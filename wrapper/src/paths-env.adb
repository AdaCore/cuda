------------------------------------------------------------------------------
--                                                                          --
--                         GNAT COMPILER COMPONENTS                         --
--                                                                          --
--                            P A T H S . E N V                             --
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

package body Paths.Env is

   ------------------------
   -- Resolve_Executable --
   ------------------------

   function Resolve_Executable (P : Path) return Path is
   begin
      if Paths.Contains_Separator (+P) then
         --  Given a "real" path:
         --  Relative paths are not accepted
         --  by OS_Lib.Locate_Exec_On_Path
         return P.Resolve;
      end if;

      --  Given just an executable name, locate it
      declare
         Ex : GNAT.OS_Lib.String_Access :=
           GNAT.OS_Lib.Locate_Exec_On_Path (+P);
         --  Will raise Access_Error in case it is not found
      begin
         return P : Path := Path (Ex.all) do
            GNAT.OS_Lib.Free (Ex);
         end return;
      end;
   end Resolve_Executable;

end Paths.Env;
