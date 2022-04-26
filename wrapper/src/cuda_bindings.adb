------------------------------------------------------------------------------
--                                                                          --
--                         GNAT COMPILER COMPONENTS                         --
--                                                                          --
--                         C U D A   B I N D I N G S                        --
--                                                                          --
--                                 B o d y                                  --
--                                                                          --
--          Copyright (C) 2010-2022, Free Software Foundation, Inc.         --
--                                                                          --
-- GNAT is free software;  you can  redistribute it  and/or modify it under --
-- terms of the  GNU General Public License as published  by the Free Soft- --
-- ware  Foundation;  either version 3,  or (at your option) any later ver- --
-- sion.  GNAT is distributed in the hope that it will be useful, but WITH- --
-- OUT ANY WARRANTY;  without even the  implied warranty of MERCHANTABILITY --
-- or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License --
-- for  more details.  You should have  received  a copy of the GNU General --
-- Public License  distributed with GNAT; see file COPYING3.  If not, go to --
-- http://www.gnu.org/licenses for a complete copy of the license.          --
--                                                                          --
-- GNAT was originally developed  by the GNAT team at  New York University. --
-- Extensive contributions were provided by Ada Core Technologies Inc.      --
--                                                                          --
------------------------------------------------------------------------------

with GNAT.OS_Lib;
use type GNAT.OS_Lib.String_Access;
with GNATVSN;
with Ada.Strings.Unbounded;
with Subprocess;  use Subprocess;
with Ada.Text_IO; use Ada.Text_IO;

package body CUDA_Bindings is

   Default_CUDA_Root : constant String := "/usr/local/cuda";
   --  !!! default not portable to Windows

   function Version_Info return String is (GNATVSN.Gnat_Static_Version_String);

   function Version_Info_Equals (Other : String) return Boolean is
      F     : Ada.Text_IO.File_Type;
      Equal : Boolean;
   begin
      if not Version_File.Is_Regular_File then
         return False;
      end if;

      Open (F, In_File, Name => +Version_File);

      if Other'Length = 0 then
         -- Empty file matches the empty string
         Equal := End_Of_File (F);
      else
         Equal := not End_Of_File (F);

         declare
            J : Natural := Other'First;
            C : Character;
         begin
            while Equal and not End_Of_File (F) loop
               if J > Other'Last then
                  -- maybe too large
                  Equal := End_Of_File (F);
               elsif End_Of_File (F) then
                  -- too short
                  Equal := False;
               else
                  Get_Immediate (F, C);
                  Equal := C = Other (J);
               end if;

               J := J + 1;
            end loop;
         end;
      end if;

      Close (F);

      return Equal;
   end Version_Info_Equals;

   procedure Write_Version_Info (Version : String) is
      F : Ada.Text_IO.File_Type;
   begin
      Create (F, Out_File, +Version_File);
      Put (F, Version);
      Close (F);
   end Write_Version_Info;

   function Up_To_Date return Boolean is (Version_Info_Equals (Version_Info));

   function Getenv_Safe (Name : String) return String is
      -- Memory-safe getenv
      S_Acc : GNAT.OS_Lib.String_Access := GNAT.OS_Lib.Getenv (Name);
      S : String := S_Acc.all;
   begin
      GNAT.OS_Lib.Free (S_Acc);
      return S;
   end Getenv_Safe;
   Getenv_Null : constant String := "";

   procedure Generate is
      Version : String := Version_Info;
   begin
      if Getenv_Safe ("CUDA_ROOT") = Getenv_Null then
         GNAT.Os_Lib.Setenv ("CUDA_ROOT", Default_CUDA_Root);
      end if;

      --  Sanity check that we're installed
      pragma Assert
        (Directory.Is_Directory,
         "directory " & (+Directory) & " doesn't exist: are we installed?");

      pragma Assert
        (Shell
           (GNAT.Os_Lib.Getenv ("SHELL").all & " bind.sh", CWD => Directory) =
         0);
      Write_Version_Info (Version);
   end Generate;

end CUDA_Bindings;
