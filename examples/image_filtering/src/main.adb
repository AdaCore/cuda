------------------------------------------------------------------------------
--                       Copyright (C) 2017, AdaCore                        --
-- This Is Free Software;  You Can Redistribute It  And/Or Modify It  Under --
-- Terms Of The  GNU General Public License As Published  By The Free Soft- --
-- Ware  Foundation;  Either Version 3,  Or (At Your Option) Any Later Ver- --
-- Sion.  This Software Is Distributed In The Hope  That It Will Be Useful, --
-- But WITHOUT ANY WARRANTY;  Without Even The Implied Warranty Of MERCHAN- --
-- TABILITY Or FITNESS FOR A PARTICULAR PURPOSE. See The GNU General Public --
-- License For  More Details.  You Should Have  Received  A Copy Of The GNU --
-- General  Public  License  Distributed  With  This  Software;   See  File --
-- COPYING3.  If Not, Go To Http://Www.Gnu.Org/Licenses For A Complete Copy --
-- Of The License.                                                          --
------------------------------------------------------------------------------

with System;
with Interfaces.C; use Interfaces.C;

with Ada.Numerics.Float_Random; use Ada.Numerics.Float_Random;
with Ada.Text_IO;               use Ada.Text_IO;

with CUDA.Driver_Types; use CUDA.Driver_Types;
with CUDA.Runtime_Api;  use CUDA.Runtime_Api;
with CUDA.Stddef;

with Ada.Unchecked_Deallocation;
with Ada.Numerics.Elementary_Functions; use Ada.Numerics.Elementary_Functions;

with Graphic;

with Bilateral_Host;
with Bilateral_Kernel;

with Importer;
with Exporter;

procedure Main is
   Width  : Natural;
   Height : Natural;

   package BK renames Bilateral_Kernel;
   package G renames Graphic;
   package BH renames Bilateral_Host;

   type Execution_Device is (Cpu, Gpu);
   Use_Dev : constant Execution_Device := Gpu;
begin

   Importer.Get_Image_Infos ("./data/ada_lovelace_photo.ppm", Width, Height);
   declare
      Img              : G.Image_Access := new G.Image (1 .. Width, 1 .. Height);
      Filtered_Img     : G.Image_Access := new G.Image (1 .. Width, 1 .. Height);
      Spatial_Stdev    : constant Float := 0.74;
      Color_Dist_Stdev : constant Float := 200.0;
   begin
      Importer.Fill_Image ("./data/ada_lovelace_photo.ppm", Width, Height, Img.all);

      Put_Line ("import done");

      case Use_Dev is
         when Cpu =>
            BH.Bilateral_Cpu (Host_Img          => Img, 
                              Host_Filtered_Img => Filtered_Img,
                              Width             => Width, 
                              Height            => Height,
                              Spatial_Stdev     => Spatial_Stdev,
                              Color_Dist_Stdev  => Color_Dist_Stdev);
         when Gpu =>
            BH.Bilateral_Cuda (Host_Img          => Img, 
                               Host_Filtered_Img => Filtered_Img,
                               Width             => Width, 
                               Height            => Height,
                               Spatial_Stdev     => Spatial_Stdev,
                               Color_Dist_Stdev  => Color_Dist_Stdev);
      end case;

      Put_Line ("bilateral done");

      Exporter.Write_Image ("./data/ada_lovelace_photo_bilateral.ppm", Filtered_Img.all);

      Put_Line ("export done");
   end;
end Main;
