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

with Ada.Strings;       use Ada.Strings;
with Ada.Strings.Fixed; use Ada.Strings.Fixed;
with Ada.Text_IO;       use Ada.Text_IO;

package body Exporter is

   procedure Export_Image (File_Path : String; Img : G.Image) is
      File   : File_Type;
      Width  : Natural := Img'Length (1);
      Height : Natural := Img'Length (2);
   begin
      Create (File, Out_File, File_Path);
      Put_Line (File, "P3");
      Put_Line (File, "#median filtered image");
      Put_Line (File, Trim (Width'Image, Left) & " " & Trim (Height'Image, Left));
      Put_Line (File, "255");
      for J in Img'Range (2) loop
         for I in Img'Range (1) loop
            Put_Line (File, Trim (Integer (Img (I, J).R)'Image, Left) & " " &
                            Trim (Integer (Img (I, J).G)'Image, Left) & " " &
                            Trim (Integer (Img (I, J).B)'Image, Left));
         end loop;
      end loop;
   end;

end Exporter;
