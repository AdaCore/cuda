with System;
with Interfaces.C; use Interfaces.C;

with Ada.Numerics.Float_Random; use Ada.Numerics.Float_Random;
with Ada.Text_IO;               use Ada.Text_IO;

with CUDA.Driver_Types; use CUDA.Driver_Types;
with CUDA.Runtime_Api;  use CUDA.Runtime_Api;
with CUDA.Stddef;


with Ada.Unchecked_Deallocation;
with Ada.Numerics.Elementary_Functions; use Ada.Numerics.Elementary_Functions;

with graphic;

with bilateral_host;
with bilateral_kernel;

with importer;
with exporter;

procedure Main is
   width  : Natural;
   height : Natural;

   package BK renames bilateral_kernel;
   package G renames graphic;
   package BH renames bilateral_host;

   type execution_device is (cpu, gpu);
   use_dev : execution_device := gpu;
begin

   importer.get_image_infos ("./data/ada_lovelace_photo.ppm", width, height);
   declare
      img              : G.image_access := new G.Image (1 .. width, 1 .. height);
      filtered_img     : G.image_access := new G.Image (1 .. width, 1 .. height);
      spatial_stdev    : float := 0.74;
      color_dist_stdev : float := 200.0;
   begin
      importer.fill_image ("./data/ada_lovelace_photo.ppm", width, height, img.all);

      Put_Line ("import done");

      case use_dev is
         when cpu =>
            BH.bilateral_cpu (
               host_img           => img, 
               host_filtered_img  => filtered_img,
               width              => width,
               height             => height,
               spatial_stdev      => spatial_stdev,
               color_dist_stdev   => color_dist_stdev
            );
         when gpu =>
            BH.bilateral_cuda (
               host_img           => img, 
               host_filtered_img  => filtered_img,
               width              => width,
               height             => height,
               spatial_stdev      => spatial_stdev,
               color_dist_stdev   => color_dist_stdev);
      end case;

      Put_Line ("bilateral done");

      exporter.write_image ("./data/ada_lovelace_photo_bilateral.ppm", filtered_img.all);

      Put_Line ("export done");
   end;
end Main;
