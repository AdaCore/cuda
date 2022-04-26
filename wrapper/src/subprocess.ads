------------------------------------------------------------------------------
--                                                                          --
--                         GNAT COMPILER COMPONENTS                         --
--                                                                          --
--                           S U B P R O C E S S                            --
--                                                                          --
--                                 S p e c                                  --
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
with Paths;
use type Paths.Path;

package Subprocess is

   Verbose : Boolean := False;

   function Spawn
     (S : String; Args : GNAT.OS_Lib.Argument_List) return Integer;
   --  Call GNAT.OS_Lib.Spawn, taking Verbose glob into account

   Default_Shell : constant String := "sh";

   function Shell (S : String; CWD : String := "") return Integer;
   --  Call a shell with the given command
   --  the shell comes from the POSIX $SHELL environement variable

   function Shell (S : String; CWD : Paths.Path) return Integer is
     (Shell (S, CWD => +CWD));
   --  Call a shell with the given command
   --  the shell comes from the POSIX $SHELL environement variable

end Subprocess;
