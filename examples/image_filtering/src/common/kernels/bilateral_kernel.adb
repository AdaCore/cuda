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

with Ada.Numerics; use Ada.Numerics;
with Ada.Numerics.Generic_Elementary_Functions;

with Interfaces.C;

package body Bilateral_Kernel is

   package CRA renames CUDA.Runtime_Api;

   procedure Bilateral (Img               : G.Image; 
                        Filtered_Img      : in out G.Image;
                        Width             : Integer; 
                        Height            : Integer; 
                        Spatial_Stdev     : Float;
                        Color_Dist_Stdev  : Float; 
                        I                 : Integer; 
                        J                 : Integer) is
      Kernel_Size : constant Integer := Integer (2.0 * Spatial_Stdev * 3.0);
      Half_Size   : constant Natural := (Kernel_Size - 1) / 2;

      Spatial_Variance : constant Float := Spatial_Stdev * Spatial_Stdev;
      Color_Dist_Variance : constant Float := Color_Dist_Stdev * Color_Dist_Stdev;

      Spatial_Gaussian    : Float := 0.0;
      Color_Dist_Gaussian : Float := 0.0;
      Sg_Cdg              : Float := 0.0;
      Sum_Sg_Cdg          : Float := 0.0;
      Rgb_Dist            : Float := 0.0;
      Spatial_Dist        : Float := 0.0;
      Filtered_Rgb        : G.Rgb := (0.0, 0.0, 0.0);

      package Fmath is new
         Ada.Numerics.Generic_Elementary_Functions (Float);

      function Distance_Square (X : Float; Y : Float) return Float is
      begin
         return X * X + Y * Y;
      end;

      function Compute_Spatial_Gaussian (K : Float) return Float is
      begin
         return
           (1.0 / (Fmath.Sqrt (2.0 * Ada.Numerics.Pi) * Spatial_Variance)) *
           Fmath.Exp (-0.5 * ((K * K) / Spatial_Variance));
      end;

      function Compute_Color_Dist_Gaussian (K : Float) return Float is
      begin
         return
           (1.0 / (Fmath.Sqrt (2.0 * Ada.Numerics.Pi) * Color_Dist_Stdev)) *
           Fmath.Exp (-0.5 * ((K * K) / Color_Dist_Variance));
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
               -- Compute Distances
               Spatial_Dist := Fmath.Sqrt (Distance_square (Float (X - I), Float(Y - J)));
               Rgb_Dist := Fmath.Sqrt (G.Distance_square (Img (I, J), Img (X, Y)));
               
               -- Compute Gaussians
               Spatial_Gaussian    := Compute_Spatial_Gaussian (Spatial_Dist);
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

   procedure Bilateral_CUDA (Device_Img          : G.Image; 
                             Device_Filtered_Img : in out G.Image;
                             Width               : Integer; 
                             Height              : Integer; 
                             Spatial_Stdev       : Float;
                             Color_Dist_Stdev    : Float) is
      use Interfaces.C;
      I : constant Integer := Integer (CRA.Block_Dim.X * CRA.Block_Idx.X + CRA.Thread_IDx.X);
      J : constant Integer := Integer (CRA.Block_Dim.Y * CRA.Block_IDx.Y + CRA.Thread_IDx.Y);
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
