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

with System;

with GNAT.OS_Lib;

with System.Storage_Elements;

with Ada.Text_IO;

package body Importer is

   package aio renames Ada.Text_IO;
   package sse renames System.Storage_Elements;

   function Load_QOI (abs_filename : String) return G.Image_Access is

      type Storage_Array_Access is access all sse.Storage_Array;
      type Input_Data is record
         Data : Storage_Array_Access;
         Desc : QOI.QOI_Desc;
      end record;

      use GNAT.OS_Lib;
      use sse;

      FD  : File_Descriptor;
      Ret : Integer;

      Result : Input_Data;

      function to_image return G.Image_Access is
         w      : Integer := integer (result.Desc.Width);
         h      : Integer := integer (result.Desc.Height);
         img    : G.Image_Access := new G.image (1 .. w, 1 .. h);
         idx    : sse.storage_count;
         offset : Natural;
         test   : Float;
      begin
         for j in img'Range (2) loop
            offset := (j - 1) * w;
            for i in img'Range (1) loop
               idx := sse.Storage_Count (((offset + (i - 1)) * 3) + 1);
               img (i,j) := (float(Result.Data(idx + 0)), float(Result.Data(idx + 1)), float(Result.Data(idx + 2)));
            end loop;
         end loop;
         return img;
      end to_image;

   begin

      aio.Put_Line ("LOAD_QOI : " & abs_filename);

      FD := GNAT.OS_Lib.Open_Read (abs_filename, Binary);

      if FD = Invalid_FD then
         aio.Put_Line (aio.Standard_Error, GNAT.OS_Lib.Errno_Message);
         GNAT.OS_Lib.OS_Exit (1);
      end if;

      declare
         Len : constant sse.Storage_Count := sse.Storage_Count (File_Length (FD));
         In_Data : constant Storage_Array_Access := new sse.Storage_Array (1 .. Len);
      begin
         Ret := Read (FD, In_Data.all'Address, In_Data.all'Length);

         if Ret /= In_Data'Length then
            aio.Put_Line (GNAT.OS_Lib.Errno_Message);
            GNAT.OS_Lib.OS_Exit (1);
         end if;

         Close (FD);

         QOI.Get_Desc (In_Data.all, Result.Desc);

         declare
            use sse;
            Out_Len : constant sse.Storage_Count := Result.Desc.Width * Result.Desc.Height * Result.Desc.Channels;
            Out_Data : constant Storage_Array_Access := new sse.Storage_Array (1 .. Out_Len);
            Output_Size : sse.Storage_Count;
         begin
            QOI.Decode (In_Data.all, Result.Desc, Out_Data.all, Output_Size);
            Result.Data := Out_Data;

            return to_image;
         end;
      end;
   end Load_QOI;

end Importer;
