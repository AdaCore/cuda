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

with Ada.Text_IO;
with Ada.Strings.Unbounded; use Ada.Strings.Unbounded;

with GNAT.Spitbol.Patterns;

package body Importer is

   procedure Get_Image_Infos (File_Path : String; 
                              Width     : out Positive; 
                              Height    : out Positive) is
      use GNAT.Spitbol.Patterns;
      use Ada.Text_IO;
      Input_File : File_Type;
   begin
      Width  := 1;
      Height := 1;
      Open (Input_File, In_File, File_Path);
      declare
         Magic_Number   : constant String  := Get_Line (Input_File);
         Note           : constant String  := Get_Line (Input_File);
         Natural_P      : constant Pattern := Span ("0123456789");
         W, H           : aliased VString_Var;
         Width_Height_P : constant Pattern := Pos (0) & Natural_P * W & Span (' ') & Natural_P * H;
         Width_Height   : VString_Var      := To_Unbounded_String (Get_Line (Input_File));
      begin
         if Match (Width_Height, Width_Height_P, "") then
            Width  := Natural'Value (To_String (W));
            Height := Natural'Value (To_String (H));
         end if;
      end;
      Close (Input_File);
   exception
      when Name_Error =>
         raise Bad_filename;
   end;

   procedure Import_Image (File_Path : String; 
                           Width     : Positive; 
                           Height    : Positive;
                           Img       : out G.Image) is
      use GNAT.Spitbol.Patterns;
      use Ada.Text_IO;
      Input_File : File_Type;
   begin
      Open (Input_File, In_File, File_Path);
      declare
         Color_Value   : VString_Var;
         Color_Value_P : constant Pattern := Span ("0123456789") * Color_Value;

         Magic_Number : constant String := Get_Line (Input_File);
         Note         : constant String := Get_Line (Input_File);
         Width_Height : constant String := Get_Line (Input_File);
         Max_Value    : constant String := Get_Line (Input_File);

         Component_Counter : Natural := 0;
         Done              : Boolean := False;

         Col, Row : Natural;
      begin
         while not Done loop
            Component_Counter := Component_Counter + 1;
            Col               := ((Component_Counter - 1) mod Width) + 1;
            Row               := (Component_Counter + (Width - 1)) / Width;
            for I in 1 .. 3 loop
               declare
                  Vline : VString_Var := To_Unbounded_String (Get_Line (Input_File));
               begin
                  if Match (Vline, Color_Value_P, "") then
                     null;
                     case I is
                        when 1 =>
                           Img (Col, Row).R := Float'Value (To_String (Color_Value));
                        when 2 =>
                           Img (Col, Row).G := Float'Value (To_String (Color_Value));
                        when 3 =>
                           Img (Col, Row).B := Float'Value (To_String (Color_Value));
                     end case;
                  end if;
               end;
            end loop;
            if End_Of_File (Input_File) then
               Done := True;
            end if;
         end loop;
      end;
      Close (Input_File);
   exception
      when Name_Error =>
         raise Bad_filename;
   end;

end Importer;
