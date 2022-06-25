------------------------------------------------------------------------------
--                       Copyright (C) 2017, AdaCore                        --
-- This is free software;  you can redistribute it  and/or modify it  under --
-- terms of the  GNU General Public License as published  by the Free Soft- --
-- ware  Foundation;  either version 3,  or (at your option) any later ver- --
-- sion.  This software is distributed in the hope  that it will be useful, --
-- but WITHOUT ANY WARRANTY;  without even the implied warranty of MERCHAN- --
-- TABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public --
-- License for  more details.  You should have  received  a copy of the GNU --
-- General  Public  License  distributed  with  this  software;   see  file --
-- COPYING3.  If not, go to http://www.gnu.org/licenses for a complete copy --
-- of the license.                                                          --
------------------------------------------------------------------------------

with CUDA.Runtime_Api;

with Graphic;

with Ada.Numerics; use Ada.Numerics;
with Ada.Numerics.Generic_Elementary_Functions;

package body Bilateral_Kernel is

   package G renames Graphic;
   package CRA renames CUDA.Runtime_Api;

   procedure Bilateral (Img_Addr          : System.Address; 
                        Filtered_Img_Addr : System.Address;
                        Width             : Integer; 
                        Height            : Integer; 
                        Spatial_Stdev     : Float;
                        Color_Dist_Stdev  : Float; 
                        I                 : Integer; 
                        J                 : Integer) is
      Kernel_Size : constant Integer := Integer (2.0 * Spatial_Stdev * 3.0);
      Half_Size   : constant Natural := (Kernel_Size - 1) / 2;

      Spatial_Gaussian    : Float := 0.0;
      Color_Dist_Gaussian : Float := 0.0;
      Sg_Cdg              : Float := 0.0;
      Sum_Sg_Cdg          : Float := 0.0;
      Rgb_Dist            : Float := 0.0;
      Filtered_Rgb        : G.Rgb := (0.0, 0.0, 0.0);

      Img          : G.Image (1 .. Width, 1 .. Height) with Address => Img_Addr;
      Filtered_Img : G.Image (1 .. Width, 1 .. Height) with Address => Filtered_Img_Addr;

      package Elementary_Functions is new
         Ada.Numerics.Generic_Elementary_Functions (Float);

      function Compute_Spatial_Gaussian (M : Float; N : Float) return Float is
         Spatial_Variance : constant Float := Spatial_Stdev * Spatial_Stdev;
         Two_Pi_Variance  : constant Float := 2.0 * 3.141_6 * Spatial_Variance;
         Expo             : constant Float := Elementary_Functions.Exp (-0.5 * ((M * M + N * N) / Spatial_Variance));
      begin
         return (1.0 / (Two_Pi_Variance)) * Expo;
      end;

      function Compute_Color_Dist_Gaussian (K : Float) return Float is
         Color_Dist_Variance : constant Float := Color_Dist_Stdev * Color_Dist_Stdev;
      begin
         return
           (1.0 / (Elementary_Functions.Sqrt (2.0 * 3.141_6) * Color_Dist_Stdev)) *
           Elementary_Functions.Exp (-0.5 * ((K * K) / Color_Dist_Variance));
      end;

      -- Compute Kernel Bounds
      Xb : constant Integer := I - Half_Size;
      Xe : constant Integer := I + Half_Size;
      Yb : constant Integer := J - Half_Size;
      Ye : constant Integer := J + Half_Size;

      use G;
   begin
      for X in Xb .. Xe loop
         for Y in Yb .. Ye loop
            if X >= 1 and X <= Width and Y >= 1 and Y <= Height then
               -- Compute Color Distance
               Rgb_Dist := Elementary_Functions.Sqrt (distance_square (Img (I, J), Img (X, Y)));

               -- Compute Gaussians
               Spatial_Gaussian    := Compute_Spatial_Gaussian (Float (I - X), Float (J - Y));
               Color_Dist_Gaussian := Compute_Color_Dist_Gaussian (Rgb_Dist);

               -- Multiply Gaussians
               Sg_Cdg := Spatial_Gaussian * Color_Dist_Gaussian;

               -- Accumulate
               Filtered_Rgb := Filtered_Rgb + (Img (X, Y) * Sg_Cdg);

               -- Track To Normalize Intensity
               Sum_Sg_Cdg := Sum_Sg_Cdg + Sg_Cdg;
            end if;
         end loop;
      end loop;

      if Sum_Sg_Cdg > 0.0 then
         -- Normalize Intensity
         Filtered_img (I, J) := G."/" (Filtered_Rgb, Sum_Sg_Cdg);
      else
         Filtered_img (I, J) := Img (I, J);
      end if;
   end;

   procedure Bilateral_CUDA (Device_Img          : System.Address; 
                             Device_Filtered_Img : System.Address;
                             Width               : Integer; 
                             Height              : Integer; 
                             Spatial_Stdev       : Float;
                             Color_Dist_Stdev    : Float) is
      I : constant Integer := Integer (CRA.Block_Idx.X);
      J : constant Integer := Integer (CRA.Block_Idx.Y);
   begin
      Bilateral (Device_Img, 
                 Device_Filtered_Img, 
                 Width, 
                 Height, 
                 Spatial_Stdev,
                 Color_Dist_Stdev, 
                 I, 
                 J);
   end;

end Bilateral_Kernel;
