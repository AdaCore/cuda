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
with Ada.Unchecked_Deallocation;
with Ada.Real_Time; use Ada.Real_Time;

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
                                                    (+"spatial_stdev",    +"3.4",       GLP.Use_Default, True, P.Set_Spatial_Stdev'Access),
                                                    (+"color_dist_stdev", +"100.0",     GLP.Use_Default, True, P.Set_Color_Dist_Stdev'Access),
                                                    (+"device",           +"gpu",       GLP.Use_Default, True, P.Set_Device'Access),
                                                    (+"out",              +"",          GLP.Use_Default, True, P.Set_Output_Image'Access));
   Param  : Parameters.User_Parameters;

   Original_Img : G.Image_Access;
   Filtered_Img : G.Image_Access;
   Start_Time   : Time;
   Elapsed_Time : Time_Span;

   procedure Free is new Ada.Unchecked_Deallocation (G.Image, G.Image_Access);

begin

   GLP.Parse_Command_Line (Parameters => Descriptors, Result => Param);
   Original_Img := Importer.Load_QOI (+Param.Input_Image);
   Filtered_Img := new G.Image (1 .. Original_Img'Length (1), 1 .. Original_Img'Length (2));

   Start_Time := Clock;
   
   case Param.Device is
      when P.Cpu =>
         BH.Bilateral_Cpu (Host_Img          => Original_Img.all, 
                           Host_Filtered_Img => Filtered_Img.all,
                           Width             => Original_Img'Length (1), 
                           Height            => Original_Img'Length (2),
                           Spatial_Stdev     => Param.Spatial_Stdev,
                           Color_Dist_Stdev  => Param.Color_Dist_Stdev);
      when P.Gpu =>
         BH.Bilateral_CUDA (Host_Img          => Original_Img.all, 
                            Host_Filtered_Img => Filtered_Img.all,
                            Width             => Original_Img'Length (1), 
                            Height            => Original_Img'Length (2),
                            Spatial_Stdev     => Param.Spatial_Stdev,
                            Color_Dist_Stdev  => Param.Color_Dist_Stdev);
   end case;

   Elapsed_Time := Clock - Start_Time;
   AIO.Put_Line ("Filtering time (" & Param.Device'Image & "): " & Duration'Image (To_Duration (Elapsed_Time)) & " seconds");

   Exporter.Dump_QOI (+Param.Output_Image, Filtered_Img);

   AIO.Put_Line ("Result found in " & Param.Output_Image'Image);

   Free (Original_Img);
   Free (Filtered_Img);

exception
   when Msg : GLP.Bad_Command =>
      AIO.Put_Line (File => AIO.Standard_Error,
                    Item => "Bad command line: " & E.Exception_Message (Msg));
      AIO.Put_Line ("Try: ./main in=./data/noisy_lena.qoi kernel=bilateral spatial_stdev=3.4 color_dist_stdev=100.0 device=gpu");
      Ada.Command_Line.Set_Exit_Status (Ada.Command_Line.Failure);
   when I.Bad_filename =>
      AIO.Put_Line ("Input file does not exists.");
   when P.Bad_extension =>
      AIO.Put_Line ("Only *.qoi images are supported.");
   when others =>
      AIO.Put_Line ("Others");
end Main;
