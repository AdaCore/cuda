------------------------------------------------------------------------------
--                                                                          --
--                         GNAT COMPILER COMPONENTS                         --
--                                                                          --
--                              G P R . E N V                               --
--                                                                          --
--                                 B o d y                                  --
--                                                                          --
--          Copyright (C) 2010-2022, Free Software Foundation, Inc.         --
--                                                                          --
-- This library is free software;  you can redistribute it and/or modify it --
-- under terms of the  GNU General Public License  as published by the Free --
-- Software  Foundation;  either version 3,  or (at your  option) any later --
-- version. This library is distributed in the hope that it will be useful, --
-- but WITHOUT ANY WARRANTY;  without even the implied warranty of MERCHAN- --
-- TABILITY or FITNESS FOR A PARTICULAR PURPOSE.                            --
--                                                                          --
-- As a special exception under Section 7 of GPL version 3, you are granted --
-- additional permissions described in the GCC Runtime Library Exception,   --
-- version 3.1, as published by the Free Software Foundation.               --
--                                                                          --
-- You should have received a copy of the GNU General Public License and    --
-- a copy of the GCC Runtime Library Exception along with this program;     --
-- see the files COPYING3 and COPYING.RUNTIME respectively.  If not, see    --
-- <http://www.gnu.org/licenses/>.                                          --
--                                                                          --
------------------------------------------------------------------------------

with Ada.Text_IO; use Ada.Text_IO;
with Ada.Directories;

with GNAT.Directory_Operations; use GNAT.Directory_Operations;
with GNAT.OS_Lib; use GNAT.OS_Lib;

with Paths;
use type Paths.Path;

package body GPR.Env is

   Path_Separator_Str : constant String (1 .. 1) := (1 => Path_Separator);

   function Test_Candidates (Path_Var : String; Name : String)
     return Paths.Path
     with Pre => Path_Var'Length /= 0 and Name'Length /= 0,
          Post => Test_Candidates'Result = Paths.Null_Path
            or (Test_Candidates'Result.Is_Regular_File
               and Test_Candidates'Result.Is_Absolute)
   is
      First : Positive := Path_Var'First;
      Last : Positive;
   begin
      --  Code adapted from libgpr's Find_Name_In_Path
      while First <= Path_Var'Last loop
         
         while First <= Path_Var'Last
           and then Path_Var (First) = Path_Separator
         loop
            First := First + 1;
         end loop;

         exit when First > Path_Var'Last;
         
         Last := First;
         while Last < Path_Var'Last
           and then Path_Var (Last) /= Path_Separator
         loop
            Last := Last + 1;
         end loop;

         if Path_Var (Last) = Path_Separator then
            Last := Last - 1;
         end if;
         
         declare
            Candidate : Paths.Path
              := Paths.Path (Path_Var (First .. Last)) / Paths.Path (Name);
         begin
            if Candidate.Is_Absolute then
               if Candidate.Is_Regular_File then
                  return Candidate;
               end if;
            else
               declare
                  Absolute_Candidate : Paths.Path
                    := Paths.Path (Ada.Directories.Current_Directory) / Paths.Path (Candidate);
               begin
                  if Absolute_Candidate.Is_Regular_File then
                     return Absolute_Candidate;
                  end if;
               end;
            end if;
         end;

         First := Last + 1;
      end loop;

      return Paths.Null_Path;
   end Test_Candidates;

   function Test_Candidates (File : Ada.Text_IO.File_Type; Name : String)
     return Paths.Path is
     -- Splits the given opened file into lines.
     -- If this line is the path of a directory that contains a file named Name
     -- return the absolute path to this file.
      Line : String (1 .. 10_000);
      -- 10_000 is from libgpr, if it fails there, the GPR file is thus not valid ;p
      Last : Natural;
   begin
      while not End_Of_File (File) loop
         Get_Line (File, Line, Last);

         if Last /= 0
           and then (Last = 1 or else Line (1 .. 2) /= "--")
         then
            declare
               Candidate : Paths.Path
                 := Paths.Path (Line (1 .. Last)) / Paths.Path (Name);
            begin
               if Candidate.Is_Regular_File then
                  return Candidate;
               end if;
            end;
         end if;
      end loop;

      return Paths.Null_Path;
   end Test_Candidates;

   function Locate_From_Path_File_Env
     (Env_Var : String; Name : String) return Paths.Path is
     -- Locate file called Name from the lines of a given file.
     -- The file containing the lines has its name in the environment
     -- variable Env_Var, if this is unset, the null path is returned.
      Env_Value : String_Access := Getenv (Env_Var);
      File : File_Type;
   begin
      if Env_Value.all /= "" then
         begin
             Open (File, In_File, Env_Value.all);

             declare
                Found_Candidate : Paths.Path
                  := Test_Candidates (File, Name);
             begin
                if Found_Candidate /= Paths.Null_Path then
                   Close (File);
                   Free (Env_Value);
                   return Found_Candidate;
                end if;
             end;
         exception
            when others =>
               Put_Line ("error reading " & Env_Value.all);
         end;

         if Is_Open (File) then
             Close (File);
         end if;
      end if;
      Free (Env_Value);

      return Paths.Null_Path;
   end Locate_From_Path_File_Env;

   function Locate_From_Path_Env
     (Env_Var : String; Name : String) return Paths.Path is
     -- Locate file called Name from all the potential path in the
     -- environment variable called Env_Var.
     -- If this Env_Var is unset, the null path is returned.
      Env_Value : String_Access := Getenv (Env_Var);
   begin
      if Env_Value.all /= "" then
         declare
            Found_Candidate : Paths.Path
              := Test_Candidates (Env_Value.all, Name);
         begin
            if Found_Candidate /= Paths.Null_Path then
               Free (Env_Value);
               return Found_Candidate;
            end if;
         end;
      end if;

      Free (Env_Value);
      return Paths.Null_Path;
   end Locate_From_Path_Env;

   -------------
   -- Resolve --
   -------------

   function Resolve (Name : String) return Paths.Path
   is
   begin

      declare
         Candidate : Paths.Path
           := Paths.Path (Ada.Directories.Current_Directory) / Paths.Path (Name);
      begin
         if Candidate.Is_Regular_File then
            return Candidate;
         end if;
      end;
    
      declare
         Candidate : Paths.Path
           := Locate_From_Path_File_Env ("GPR_PROJECT_PATH_FILE", Name);
      begin
         if Candidate.Is_Regular_File then
            return Candidate;
         end if;
      end;
    
      declare
         Candidate : Paths.Path
           := Locate_From_Path_Env ("GPR_PROJECT_PATH", Name);
      begin
         if Candidate.Is_Regular_File then
            return Candidate;
         end if;
      end;
    
      declare
         Candidate : Paths.Path
           := Locate_From_Path_Env ("ADA_PROJECT_PATH", Name);
      begin
         if Candidate.Is_Regular_File then
            return Candidate;
         end if;
      end;

      return Paths.Null_Path;
   end Resolve;

end GPR.Env;
