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

With CUDA.Stddef;
With CUDA.Vector_Types;
With CUDA.Driver_Types;

With Storage_Models;
With Storage_Models.Arrays;
With Storage_Models.Objects;
With CUDA_Storage_Models;

With Interfaces.C;
Use Interfaces.C; -- Operators For Block_Dim, Block_Idx, Thread_Idx

With System;

With CUDA.Runtime_Api;

With Bilateral_Kernel;

Package Body Bilateral_Host Is

    Package BK Renames Bilateral_Kernel;

    Procedure Bilateral_Cpu
       (Host_Img          : G.Image_Access; 
        Host_Filtered_Img : G.Image_Access;
        Width             : Integer; 
        Height            : Integer; 
        Spatial_Stdev     : Float;
        Color_Dist_Stdev  : Float) Is
    Begin
        For I In Host_Img.All'Range (1) Loop
            For J In Host_Img.All'Range (2) Loop
                BK.Bilateral
                   (Img_Addr          => Host_Img.All'Address,
                    Filtered_Img_Addr => Host_Filtered_Img.All'Address,
                    Width             => Width, Height => Height,
                    Spatial_Stdev     => Spatial_Stdev,
                    Color_Dist_Stdev  => Color_Dist_Stdev, I => I, J => J);
            End Loop;
        End Loop;
    End;

    Procedure Bilateral_Cuda (Host_Img          : G.Image_Access; 
                              Host_Filtered_Img : G.Image_Access;
                              Width             : Integer; 
                              Height            : Integer; 
                              Spatial_Stdev     : Float;
                              Color_Dist_Stdev  : Float) Is
        Image_Bytes       : CUDA.Stddef.Size_T              := CUDA.Stddef.Size_T (Host_Img.All'Size / 8);
        Threads_Per_Block : Constant Cuda.Vector_Types.Dim3 := (1, 1, 1);
        Blocks_Per_Grid   : Constant Cuda.Vector_Types.Dim3 := (Interfaces.C.Unsigned (Width), Interfaces.C.Unsigned (Height), 1);

        Device_Img, Device_Filtered_Img : System.Address;
    Begin

        Device_Img := CUDA.Runtime_Api.Malloc (Image_Bytes);
        CUDA.Runtime_Api.Memcpy (Device_Img, Host_Img.All'Address, Image_Bytes, CUDA.Driver_Types.Memcpy_Host_To_Device);

        Device_Filtered_Img := CUDA.Runtime_Api.Malloc (Image_Bytes);
        CUDA.Runtime_Api.Memcpy (Device_Filtered_Img, Host_Filtered_Img.All'Address, Image_Bytes, CUDA.Driver_Types.Memcpy_Host_To_Device);

        Pragma Cuda_Execute (BK.Bilateral_Cuda (Device_Img, 
                                                Device_Filtered_Img, 
                                                Width, 
                                                Height, 
                                                Spatial_Stdev,
                                                Color_Dist_Stdev),
                             Blocks_Per_Grid, 
                             Threads_Per_Block);

        CUDA.Runtime_Api.Memcpy (Host_Filtered_Img.All'Address, Device_Filtered_Img, Image_Bytes, CUDA.Driver_Types.Memcpy_Device_To_Host);
    End;
End Bilateral_Host;
