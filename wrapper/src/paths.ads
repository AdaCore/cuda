------------------------------------------------------------------------------
--                                                                          --
--                         GNAT COMPILER COMPONENTS                         --
--                                                                          --
--                                P A T H S                                 --
--                                                                          --
--                                 S p e c                                  --
--                                                                          --
--          Copyright (C) 1992-2022, Free Software Foundation, Inc.         --
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

with Ada.Command_Line;
with GNAT.OS_Lib;
with Ada.Directories;
with Ada.Strings;       use Ada.Strings;
with Ada.Strings.Fixed; use Ada.Strings.Fixed;
with Ada.Strings.Unbounded;

package Paths is
   type Path is new String;

   function "+" (P : Path) return String is (String (P));

   function "+" (S : Ada.Strings.Unbounded.Unbounded_String) return Path is
     (Path (Ada.Strings.Unbounded.To_String (S)));

   function Is_Absolute (P : Path) return Boolean is
     (GNAT.OS_Lib.Is_Absolute_Path (+P));

   function Is_Directory (P : Path) return Boolean is
     (GNAT.OS_Lib.Is_Directory (+P));

   function Is_Regular_File (P : Path) return Boolean is
     (GNAT.OS_Lib.Is_Regular_File (+P));

   function Exists (P : Path) return Boolean is (Ada.Directories.Exists (+P));

   function "/" (P1 : Path; P2 : Path) return Path is
     (P1 & GNAT.OS_Lib.Directory_Separator & P2) with
      Pre => not Is_Absolute (P2);

   Directory_Separator_Str : constant String (1 .. 1) :=
     (1 => GNAT.OS_Lib.Directory_Separator);

   function Contains_Separator (S : String) return Boolean is
     (Index (S, Directory_Separator_Str) /= 0);

   function Parent (P : Path) return Path is
     (P
        (P'First ..
             Index (+P, Directory_Separator_Str, Going => Backward) - 1)) with
      Pre => Contains_Separator
        (+P) -- no path separator, no parent
      and P'Length > 1; --  single path separator, no parent

   function Resolve (P : Path) return Path is
     (Path (GNAT.OS_Lib.Normalize_Pathname (+P)));

end Paths;
