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

With Ada.Strings;       Use Ada.Strings;
With Ada.Strings.Fixed; Use Ada.Strings.Fixed;
With Ada.Text_IO;       Use Ada.Text_IO;

Package Body Exporter Is

    Procedure Write_Image (File_Path : String; Img : G.Image) Is
        File   : File_Type;
        Width  : Natural := Img'Length (1);
        Height : Natural := Img'Length (2);
    Begin
        Create (File, Out_File, File_Path);
        Put_Line (File, "P3");
        Put_Line (File, "#median filtered image");
        Put_Line (File, Trim (Width'Image, Left) & " " & Trim (Height'Image, Left));
        Put_Line (File, "255");
        For J In Img'Range (2) Loop
            For I In Img'Range (1) Loop
                Put_Line (File, Trim (Integer(Img (I, J).R)'Image, Left) & " " & 
                                Trim (Integer(Img (I, J).G)'Image, Left) & " " & 
                                Trim (Integer(Img (I, J).B)'Image, Left));
            End Loop;
        End Loop;
    End;

End Exporter;
