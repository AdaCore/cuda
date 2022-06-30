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

<<<<<<< HEAD
with Qoi;
=======
with QOI;
>>>>>>> changes to make it work

with System;

with GNAT.OS_Lib;

with System.Storage_Elements;

with Ada.Text_IO;

package body Importer is

   package Aio renames Ada.Text_IO;
   package Sse renames System.Storage_Elements;

   function Load_Qoi (Abs_Filename : String) return G.Image_Access is

      type Storage_Array_Access is access all Sse.Storage_Array;
      type Input_Data is record
         Data : Storage_Array_Access;
         Desc : Qoi.Qoi_Desc;
      end record;

      use GNAT.OS_Lib;
      use Sse;

      Fd  : File_Descriptor;
      Ret : Integer;

      Result : Input_Data;

      function To_Image return G.Image_Access is
         W      : Integer        := Integer (Result.Desc.Width);
         H      : Integer        := Integer (Result.Desc.Height);
         Img    : G.Image_Access := new G.Image (1 .. W, 1 .. H);
         Idx    : Sse.Storage_Count;
         Offset : Natural;
         Test   : Float;
      begin
         for J in Img'Range (2) loop
            Offset := (J - 1) * W;
            for I in Img'Range (1) loop
               Idx        := Sse.Storage_Count (((Offset + (I - 1)) * 3) + 1);
               Img (I, J) :=
                 (Float (Result.Data (Idx + 0)), Float (Result.Data (Idx + 1)),
                  Float (Result.Data (Idx + 2)));
            end loop;
         end loop;
         return Img;
      end To_Image;

   begin

      Aio.Put_Line ("LOAD_QOI : " & Abs_Filename);

      Fd := GNAT.OS_Lib.Open_Read (Abs_Filename, Binary);

      if Fd = Invalid_FD then
         Aio.Put_Line (Aio.Standard_Error, GNAT.OS_Lib.Errno_Message);
         GNAT.OS_Lib.OS_Exit (1);
      end if;

      declare
         Len : constant Sse.Storage_Count :=
           Sse.Storage_Count (File_Length (Fd));
         In_Data : constant Storage_Array_Access :=
           new Sse.Storage_Array (1 .. Len);
      begin
         Ret := Read (Fd, In_Data.all'Address, In_Data.all'Length);

         if Ret /= In_Data'Length then
            Aio.Put_Line (GNAT.OS_Lib.Errno_Message);
            GNAT.OS_Lib.OS_Exit (1);
         end if;

         Close (Fd);

         Qoi.Get_Desc (In_Data.all, Result.Desc);

         declare
            use Sse;
            Out_Len : constant Sse.Storage_Count :=
              Result.Desc.Width * Result.Desc.Height * Result.Desc.Channels;
            Out_Data : constant Storage_Array_Access :=
              new Sse.Storage_Array (1 .. Out_Len);
            Output_Size : Sse.Storage_Count;
         begin
            Qoi.Decode (In_Data.all, Result.Desc, Out_Data.all, Output_Size);
            Result.Data := Out_Data;

            return To_Image;
         end;
      end;
   end Load_Qoi;

end Importer;
