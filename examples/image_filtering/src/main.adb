With System;
With Interfaces.C; Use Interfaces.C;

With Ada.Numerics.Float_Random; Use Ada.Numerics.Float_Random;
With Ada.Text_IO;               Use Ada.Text_IO;

With CUDA.Driver_Types; Use CUDA.Driver_Types;
With CUDA.Runtime_Api;  Use CUDA.Runtime_Api;
With CUDA.Stddef;

With Ada.Unchecked_Deallocation;
With Ada.Numerics.Elementary_Functions; Use Ada.Numerics.Elementary_Functions;

With Graphic;

With Bilateral_Host;
With Bilateral_Kernel;

With Importer;
With Exporter;

Procedure Main Is
   Width  : Natural;
   Height : Natural;

   Package BK Renames Bilateral_Kernel;
   Package G Renames Graphic;
   Package BH Renames Bilateral_Host;

   Type Execution_Device Is (Cpu, Gpu);
   Use_Dev : Execution_Device := Gpu;
Begin

   Importer.Get_Image_Infos ("./data/ada_lovelace_photo.ppm", Width, Height);
   Declare
      Img              : G.Image_Access := New G.Image (1 .. Width, 1 .. Height);
      Filtered_Img     : G.Image_Access := New G.Image (1 .. Width, 1 .. Height);
      Spatial_Stdev    : Float := 0.74;
      Color_Dist_Stdev : Float := 200.0;
   Begin
      Importer.Fill_Image ("./data/ada_lovelace_photo.ppm", Width, Height, Img.All);

      Put_Line ("import done");

      Case Use_Dev Is
         When Cpu =>
            BH.Bilateral_Cpu (
               Host_Img           => Img, 
               Host_Filtered_Img  => Filtered_Img,
               Width              => Width,
               Height             => Height,
               Spatial_Stdev      => Spatial_Stdev,
               Color_Dist_Stdev   => Color_Dist_Stdev
            );
         When Gpu =>
            BH.Bilateral_Cuda (
               Host_Img           => Img, 
               Host_Filtered_Img  => Filtered_Img,
               Width              => Width,
               Height             => Height,
               Spatial_Stdev      => Spatial_Stdev,
               Color_Dist_Stdev   => Color_Dist_Stdev);
      End Case;

      Put_Line ("bilateral done");

      Exporter.Write_Image ("./data/ada_lovelace_photo_bilateral.ppm", Filtered_Img.All);

      Put_Line ("export done");
   End;
End Main;
