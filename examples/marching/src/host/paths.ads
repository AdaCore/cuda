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

   function Root_Dir return Path
   --  Root dir of the project, based on the fact that main is placed in it
   is
     (Parent
        (Path
           (GNAT.OS_Lib.Normalize_Pathname
              (Ada.Command_Line.Command_Name)))) with
     Pre  => Contains_Separator (Ada.Command_Line.Command_Name),
     Post => Is_Absolute (Root_Dir'Result) and Is_Directory (Root_Dir'Result);

   function Resolve_From_Root (File : Path) return Path is
     (if Is_Absolute (File) then File else Root_Dir / File) with
     Post =>
      Is_Absolute (Resolve_From_Root'Result) and
      Exists (Resolve_From_Root'Result);

end Paths;
