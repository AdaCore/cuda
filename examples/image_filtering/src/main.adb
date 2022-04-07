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

with Ada.Text_IO;
with Ada.Strings.Unbounded; use Ada.Strings.Unbounded;
with Ada.Command_Line;
with Ada.Exceptions;

with Generic_Line_Parser;

with Graphic;

with Bilateral_Host;
with Bilateral_Kernel;

with Importer;
with Exporter;
with Parameters;

procedure Main is
   package AIO renames Ada.Text_IO;
   package E   renames Ada.Exceptions;
   package ASU renames Ada.Strings.Unbounded;
   package G   renames Graphic;
   package BK  renames Bilateral_Kernel;
   package BH  renames Bilateral_Host;
   package I   renames Importer;
   package P   renames Parameters;

   package GLP is new Generic_Line_Parser (Parameters.User_Parameters);

   function "+" (Str : String) return ASU.Unbounded_String is (ASU.To_Unbounded_String (Str));
   function "+" (Str : ASU.Unbounded_String) return String is (ASU.To_String (Str));

   Descriptors : GLP.Parameter_Descriptor_Array := ((+"in",               +"",          GLP.Die,         True, P.Set_Input_Image'Access),
                                                    (+"kernel",           +"bilateral", GLP.Use_Default, True, P.Set_Kernel'Access),
                                                    (+"spatial_stdev",    +"0.75",      GLP.Use_Default, True, P.Set_Spatial_Stdev'Access),
                                                    (+"color_dist_stdev", +"120.0",     GLP.Use_Default, True, P.Set_Color_Dist_Stdev'Access),
                                                    (+"device",           +"gpu",       GLP.Use_Default, True, P.Set_Device'Access),
                                                    (+"out",              +"",          GLP.Use_Default, True, P.Set_Output_Image'Access));
   Param  : Parameters.User_Parameters;
   Width  : Natural;
   Height : Natural;
begin
   GLP.Parse_Command_Line (Parameters => Descriptors, Result => Param);
   
   Importer.Get_Image_Infos (+Param.Input_Image, Width, Height);

   declare
      Img              : G.Image_Access := new G.Image (1 .. Width, 1 .. Height);
      Filtered_Img     : G.Image_Access := new G.Image (1 .. Width, 1 .. Height);
   begin
      Importer.Import_Image (+Param.Input_Image, Width, Height, Img.all);
      AIO.Put_Line ("Import done");

      case Param.Device is
         when P.Cpu =>
            BH.Bilateral_Cpu (Host_Img          => Img, 
                              Host_Filtered_Img => Filtered_Img,
                              Width             => Width, 
                              Height            => Height,
                              Spatial_Stdev     => Param.Spatial_Stdev,
                              Color_Dist_Stdev  => Param.Color_Dist_Stdev);
         when P.Gpu =>
            BH.Bilateral_Cuda (Host_Img          => Img, 
                               Host_Filtered_Img => Filtered_Img,
                               Width             => Width, 
                               Height            => Height,
                               Spatial_Stdev     => Param.Spatial_Stdev,
                               Color_Dist_Stdev  => Param.Color_Dist_Stdev);
      end case;
      AIO.Put_Line ("Filtering done");

      Exporter.Export_Image (+Param.Output_Image, Filtered_Img.all);
      AIO.Put_Line ("Export done");

      AIO.Put_Line ("Result found in " & Param.Output_Image'Image);

      G.Free (Img);
      G.Free (Filtered_Img);
   end;
exception
   when Msg : GLP.Bad_Command =>
      Ada.Text_IO.Put_Line (File => Ada.Text_IO.Standard_Error,
                            Item => "Bad command line: " & E.Exception_Message (Msg));
      Ada.Text_IO.Put_Line ("Try: ./img_filter in=./data/ada_lovelace_photo.ppm kernel=bilateral spatial_stdev=0.75 color_dist_stdev=120.0 device=gpu");
      Ada.Command_Line.Set_Exit_Status (Ada.Command_Line.Failure);
   when I.Bad_filename =>
      Ada.Text_IO.Put_Line ("Input file does not exists.");
   when P.Bad_extension =>
      Ada.Text_IO.Put_Line ("Only *.ppm images are supported.");
end Main;
