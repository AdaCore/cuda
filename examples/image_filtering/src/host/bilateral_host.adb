with CUDA.Stddef;
with CUDA.Vector_Types;
with CUDA.Driver_Types;

with Storage_Models;
with Storage_Models.Arrays;
with Storage_Models.Objects;
with CUDA_Storage_Models;

with Interfaces.C;
use Interfaces.C; -- operators for block_dim, block_idx, thread_idx

with System;

with CUDA.Runtime_Api;

with bilateral_kernel;

package body bilateral_host is

    package BK renames bilateral_kernel;

    procedure bilateral_cpu
       (host_img          : G.image_access; 
        host_filtered_img : G.image_access;
        width             : Integer; 
        height            : Integer; 
        spatial_stdev     : Float;
        color_dist_stdev  : Float) is
    begin
        for i in host_img.all'Range (1) loop
            for j in host_img.all'Range (2) loop
                BK.bilateral
                   (img_addr          => host_img.all'Address,
                    filtered_img_addr => host_filtered_img.all'Address,
                    width             => width, height => height,
                    spatial_stdev     => spatial_stdev,
                    color_dist_stdev  => color_dist_stdev, i => i, j => j);
            end loop;
        end loop;
    end;

    procedure bilateral_cuda (host_img          : G.image_access; 
                              host_filtered_img : G.image_access;
                              width             : Integer; 
                              height            : Integer; 
                              spatial_stdev     : Float;
                              color_dist_stdev  : Float) is
        image_bytes       : CUDA.Stddef.Size_T              := CUDA.Stddef.Size_T (host_img.all'Size / 8);
        threads_per_block : constant cuda.vector_types.dim3 := (1, 1, 1);
        blocks_per_grid   : constant cuda.vector_types.dim3 := (interfaces.c.unsigned (width), interfaces.c.unsigned (height), 1);

        device_img, device_filtered_img : System.Address;
    begin

        device_img := CUDA.Runtime_Api.Malloc (image_bytes);
        CUDA.Runtime_Api.Memcpy (device_img, host_img.all'Address, image_bytes, CUDA.Driver_Types.Memcpy_Host_To_Device);

        device_filtered_img := CUDA.Runtime_Api.Malloc (image_bytes);
        CUDA.Runtime_Api.Memcpy (device_filtered_img, host_filtered_img.all'Address, image_bytes, CUDA.Driver_Types.Memcpy_Host_To_Device);

        pragma Cuda_Execute (BK.bilateral_cuda (device_img, 
                                                device_filtered_img, 
                                                width, 
                                                height, 
                                                spatial_stdev,
                                                color_dist_stdev),
                             blocks_per_grid, 
                             threads_per_block);

        CUDA.Runtime_Api.Memcpy (host_filtered_img.all'Address, device_filtered_img, image_bytes, CUDA.Driver_Types.Memcpy_Device_To_Host);
    end;
end bilateral_host;
