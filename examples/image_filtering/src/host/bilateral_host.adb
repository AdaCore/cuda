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

   procedure Bilateral_Cpu (Host_Img          : G.Image_Access; 
                            Host_Filtered_Img : G.Image_Access;
                            Width             : Integer; 
                            Height            : Integer; 
                            Spatial_Stdev     : Float;
                            Color_Dist_Stdev  : Float) is
   begin
      for I in Host_Img.all'Range (1) loop
         for J in Host_Img.all'Range (2) loop
            BK.Bilateral (Img_Addr          => Host_Img.all'Address,
                          Filtered_Img_Addr => Host_Filtered_Img.all'Address,
                          Width             => Width, Height => Height,
                          Spatial_Stdev     => Spatial_Stdev,
                          Color_Dist_Stdev  => Color_Dist_Stdev, 
                          I                 => I, 
                          J                 => J);
         end loop;
      end loop;
   end;

   procedure Bilateral_CUDA (Host_Img          : G.Image_Access; 
                             Host_Filtered_Img : G.Image_Access;
                             Width             : Integer; 
                             Height            : Integer; 
                             Spatial_Stdev     : Float;
                             Color_Dist_Stdev  : Float) is
      Image_Bytes       : constant CUDA.Stddef.Size_T     := CUDA.Stddef.Size_T (Host_Img.all'Size / 8);
      Threads_Per_Block : constant CUDA.Vector_Types.Dim3 := (1, 1, 1);
      Blocks_Per_Grid   : constant CUDA.Vector_Types.Dim3 := (IC.Unsigned (Width), 
                                                              IC.Unsigned (Height), 
                                                              1);
      Device_Img, Device_Filtered_Img : System.Address;
   begin
      -- send input data to device
      Device_Img := CRA.Malloc (Image_Bytes);
      CRA.Memcpy (Device_Img, 
                  Host_Img.all'Address, 
                  Image_Bytes, 
                  CDT.Memcpy_Host_To_Device);

      Device_Filtered_Img := CRA.Malloc (Image_Bytes);
      CRA.Memcpy (Device_Filtered_Img, 
                  Host_Filtered_Img.all'Address, 
                  Image_Bytes, 
                  CDT.Memcpy_Host_To_Device);

      -- compute filter kernel on device
      pragma CUDA_Execute (BK.Bilateral_CUDA (Device_Img, 
                                              Device_Filtered_Img, 
                                              Width, 
                                              Height, 
                                              Spatial_Stdev,
                                              Color_Dist_Stdev),
                           Blocks_Per_Grid, 
                           Threads_Per_Block);

      -- send output data to host
      CRA.Memcpy (Host_Filtered_Img.all'Address, 
                  Device_Filtered_Img, 
                  Image_Bytes,
                  CDT.Memcpy_Device_To_Host);

      CRA.Free (Device_Img);
      CRA.Free (Device_Filtered_Img);
   end;
end Bilateral_Host;
