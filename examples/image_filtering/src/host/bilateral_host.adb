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

with CUDA.Stddef;
with CUDA.Vector_Types;
with CUDA.Driver_Types;

with Storage_Models;
with Storage_Models.Arrays;
with Storage_Models.Objects;
with CUDA_Storage_Models;

with Interfaces.C;
use Interfaces.C; -- Operators For Block_Dim, Block_Idx, Thread_Idx

with System;

with CUDA.Runtime_Api;

with Bilateral_Kernel;

package body Bilateral_Host is

   package BK renames Bilateral_Kernel;
   package CDT renames CUDA.Driver_Types;

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

   procedure Bilateral_Cuda (Host_Img          : G.Image_Access; 
                             Host_Filtered_Img : G.Image_Access;
                             Width             : Integer; 
                             Height            : Integer; 
                             Spatial_Stdev     : Float;
                             Color_Dist_Stdev  : Float) is
      Image_Bytes       : constant CUDA.Stddef.Size_T     := CUDA.Stddef.Size_T (Host_Img.all'Size / 8);
      Threads_Per_Block : constant Cuda.Vector_Types.Dim3 := (1, 1, 1);
      Blocks_Per_Grid   : constant Cuda.Vector_Types.Dim3 := (Interfaces.C.Unsigned (Width), 
                                                              Interfaces.C.Unsigned (Height), 
                                                              1);
      Device_Img, Device_Filtered_Img : System.Address;
   begin
      -- send input data to device
      Device_Img := CUDA.Runtime_Api.Malloc (Image_Bytes);
      CUDA.Runtime_Api.Memcpy (Device_Img, 
                               Host_Img.all'Address, 
                               Image_Bytes, 
                               CDT.Memcpy_Host_To_Device);

      Device_Filtered_Img := CUDA.Runtime_Api.Malloc (Image_Bytes);
      CUDA.Runtime_Api.Memcpy (Device_Filtered_Img, 
                               Host_Filtered_Img.all'Address, 
                               Image_Bytes, 
                               CDT.Memcpy_Host_To_Device);

      -- compute filter kernel on device
      pragma Cuda_Execute (BK.Bilateral_Cuda (Device_Img, 
                                              Device_Filtered_Img, 
                                              Width, 
                                              Height, 
                                              Spatial_Stdev,
                                              Color_Dist_Stdev),
                           Blocks_Per_Grid, 
                           Threads_Per_Block);

      -- send output data to host
      CUDA.Runtime_Api.Memcpy (Host_Filtered_Img.all'Address, 
                               Device_Filtered_Img, 
                               Image_Bytes,
                               CDT.Memcpy_Device_To_Host);

      -- declare
      --    Img                   : aliased G.Image (1 .. Width, 1 .. Height) with Address => Device_Img;
      --    Img_Access            : G.Image_Access := Img'Unrestricted_Access;
      --    Filtered_Img          : aliased G.Image (1 .. Width, 1 .. Height) with Address => Device_Filtered_Img;
      --    Filtered_Img_Access   : G.Image_Access := Filtered_Img'Unrestricted_Access;
      -- begin
      --    G.Free (Img_Access);
      --    G.Free (Filtered_Img_Access);
      -- end;
   end;
end Bilateral_Host;
