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

with Ada.Unchecked_Deallocation;

with Interfaces.C;
with System;

with CUDA.Stddef;
with CUDA.Vector_Types;
with CUDA.Driver_Types;
with CUDA.Runtime_Api;

with Bilateral_Kernel;

package body Bilateral_Host is

   package BK renames Bilateral_Kernel;
   package CDT renames CUDA.Driver_Types;
   package CRA renames CUDA.Runtime_Api;
   package IC renames Interfaces.C;

   procedure Bilateral_Cpu (Host_Img          : G.Image; 
                            Host_Filtered_Img : in out G.Image;
                            Width             : Integer; 
                            Height            : Integer; 
                            Spatial_Stdev     : Float;
                            Color_Dist_Stdev  : Float) is
   begin
      for I in Host_Img'Range (1) loop
         for J in Host_Img'Range (2) loop
            BK.Bilateral (Img               => Host_Img,
                          Filtered_Img      => Host_Filtered_Img,
                          Width             => Width, Height => Height,
                          Spatial_Stdev     => Spatial_Stdev,
                          Color_Dist_Stdev  => Color_Dist_Stdev, 
                          I                 => I, 
                          J                 => J);
         end loop;
      end loop;
   end;

   procedure Bilateral_CUDA (Host_Img          : G.Image;
                             Host_Filtered_Img : in out G.Image;
                             Width             : Integer; 
                             Height            : Integer; 
                             Spatial_Stdev     : Float;
                             Color_Dist_Stdev  : Float) is
      Image_Bytes : constant CUDA.Stddef.Size_T := CUDA.Stddef.Size_T (Host_Img'Size / 8);

      procedure Free is new Ada.Unchecked_Deallocation (G.Image, G.Image_Device_Access);

      use IC;
      Threads_Per_Block : constant CUDA.Vector_Types.Dim3 := (16, 16, 1);
      Block_X : constant IC.Unsigned := (IC.Unsigned (Width) + Threads_Per_Block.X - 1) / Threads_Per_Block.X;
      Block_Y : constant IC.Unsigned := (IC.Unsigned (Height) + Threads_Per_Block.Y - 1) / Threads_Per_Block.Y;
      Blocks_Per_Grid : constant CUDA.Vector_Types.Dim3 := (Block_X, Block_Y, 1);

      -- data to device
      Device_Img : G.Image_Device_Access := new G.Image'(Host_Img);
      Device_Filtered_Img : G.Image_Device_Access := new G.Image'(Host_Filtered_Img);
   begin

      -- compute filter kernel on device
      pragma CUDA_Execute (BK.Bilateral_CUDA (Device_Img, 
                                              Device_Filtered_Img, 
                                              Width, 
                                              Height, 
                                              Spatial_Stdev,
                                              Color_Dist_Stdev),
                           Blocks_Per_Grid, 
                           Threads_Per_Block);

      -- data to host
      Host_Filtered_Img := Device_Filtered_Img.all;

      Free (Device_Img);
      Free (Device_Filtered_Img);
   end;
end Bilateral_Host;
