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

with QOI;
with System.Storage_Elements;

with Ada.Text_IO; use Ada.Text_IO;

with GNAT.OS_Lib;

package body Exporter is

   package sse renames System.Storage_Elements;

   procedure Write_To_File (Filename : String;
                            D : sse.Storage_Array;
                            Size : sse.Storage_Count)
   is
      use GNAT.OS_Lib;

      FD : File_Descriptor;
      Ret : Integer;
   begin

      FD := GNAT.OS_Lib.Create_File (Filename, Binary);

      if FD = Invalid_FD then
         Ada.Text_IO.Put_Line (GNAT.OS_Lib.Errno_Message);
         GNAT.OS_Lib.OS_Exit (1);
      end if;

      Ret := Write (FD, D'Address, Integer (Size));

      if Ret /= Integer (Size) then
         Ada.Text_IO.Put_Line (GNAT.OS_Lib.Errno_Message);
         GNAT.OS_Lib.OS_Exit (1);
      end if;

      Close (FD);
   end Write_To_File;

   procedure Dump_QOI (abs_filename : String; Img : G.Image_Access) is
   use sse;
      Desc : QOI.QOI_Desc := (Img'Length(1), Img'Length(2), 3, QOI.SRGB);
      
      type Storage_Array_Access is access all sse.Storage_Array;

      function to_storage return Storage_Array_Access is
         data  : Storage_Array_Access := 
            new sse.Storage_Array (1 .. Desc.Width * Desc.Height * Desc.Channels);
         j_offset : Natural;
         i_offset : Natural;
         t_offset : Natural;
      begin
         for j in img'Range (2) loop
            j_offset := (j-1) * Natural(Desc.Width * Desc.Channels);
            for i in img'Range (1) loop
               i_offset := (i-1) * Natural(Desc.Channels);
               t_offset := j_offset + i_offset;
               data (sse.Storage_Offset(t_offset + 1)) := sse.Storage_Element(Img(i,j).R);
               data (sse.Storage_Offset(t_offset + 2)) := sse.Storage_Element(Img(i,j).G);
               data (sse.Storage_Offset(t_offset + 3)) := sse.Storage_Element(Img(i,j).B);
            end loop;
         end loop;
         return data;
      end;

      Data        : Storage_Array_Access := to_storage;
      Output      : Storage_Array_Access := new sse.Storage_Array (1 .. QOI.Encode_Worst_Case (Desc));
      Output_Size : sse.Storage_Count;
   begin
      QOI.Encode (Data.all, Desc, Output.all, Output_Size);

      if Output_Size /= 0 then
         Put_Line ("DUMP_QOI : " & abs_filename);
         Write_To_File (abs_filename, Output.all, Output_Size);
      else
         Put_Line ("DUMP_QOI : Encode failed");
         GNAT.OS_Lib.OS_Exit (1);
      end if;
   end;
end Exporter;
