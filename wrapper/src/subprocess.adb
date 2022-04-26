------------------------------------------------------------------------------
--                                                                          --
--                         GNAT COMPILER COMPONENTS                         --
--                                                                          --
--                           S U B P R O C E S S                            --
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
with GNAT;
with GNAT.OS_Lib; use GNAT.OS_Lib;
with Ada.Text_IO; use Ada.Text_IO;
with Ada.Directories;

package body Subprocess is

   -----------
   -- Spawn --
   -----------

   function Spawn (S : String; Args : Argument_List) return Integer is
   begin
      if Verbose then
         Put ("+ " & S);

         for J in Args'Range loop
            Put (" " & Args (J).all);
         end loop;

         New_Line;
      end if;

      return GNAT.OS_Lib.Spawn (S, Args);
   end Spawn;

   ----------------------
   -- Shell_Executable --
   ----------------------

   function Shell_Executable return String is
      S_Acc : String_Access := Getenv ("SHELL");
   begin
      if S_Acc /= null then
         return S : String := S_Acc.all do
            Free (S_Acc);
         end return;
      else
         return Default_Shell;
      end if;
   end Shell_Executable;

   -----------
   -- Shell --
   -----------

   function Shell (S : String; CWD : String := "") return Integer is

      Ret        : Integer;
      Prev_Dir   : String           := Ada.Directories.Current_Directory;
      Change_Dir : constant Boolean := (CWD /= "");

      Opt  : aliased String         := "-c";
      Cmd  : aliased String         := S;
      Args : constant Argument_List :=
        (Opt'Unchecked_Access, Cmd'Unchecked_Access);

   begin
      if Change_Dir then
         Ada.Directories.Set_Directory (CWD);
         if Verbose then
            Put_Line ("entering " & CWD);
         end if;
      end if;

      Ret := Spawn (Shell_Executable, Args);

      if Change_Dir then
         Ada.Directories.Set_Directory (Prev_Dir);
         if Verbose then
            Put_Line ("leaving " & CWD);
         end if;
      end if;

      return Ret;
   end Shell;

end Subprocess;
