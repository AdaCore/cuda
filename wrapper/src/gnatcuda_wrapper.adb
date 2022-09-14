------------------------------------------------------------------------------
--                                                                          --
--                         GNAT COMPILER COMPONENTS                         --
--                                                                          --
--                     G N A T C U D A _ W R A P P E R                      --
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

with Ada.Strings.Unbounded;     use Ada.Strings.Unbounded;
with Ada.Command_Line;          use Ada.Command_Line;
with Ada.Environment_Variables; use Ada.Environment_Variables;

with GNAT.Case_Util;            use GNAT.Case_Util;
with GNAT.Directory_Operations; use GNAT.Directory_Operations;
with GNAT.IO;                   use GNAT.IO;
with GNAT.IO_Aux;               use GNAT.IO_Aux;
with GNAT.OS_Lib;               use GNAT.OS_Lib;

with Gnatvsn;

--  Wrapper around <install>/libexec/gnat_ccg/bin/c-xxx to be
--  installed under <install>/bin

function GNATCUDA_Wrapper return Integer is

   subtype String_Access is GNAT.OS_Lib.String_Access;
   --  GNAT.OS_Lib and Ada.Strings.Unbounded both declare String_Access, we care
   --  about the one for OS_Lib here.

   Exec_Not_Found : exception;

   function Executable_Location return String;
   --  Return the name of the parent directory where the executable is stored
   --  (so if you are running "prefix"/bin/gcc, you would get "prefix").
   --  A special case is done for "bin" directories, which are skipped.
   --  The returned directory always ends up with a directory separator.

   function Is_Directory_Separator (C : Character) return Boolean;
   --  Return True if C is a directory separator

   --  function Locate_Exec (Exec : String) return String;
   --  Locate Exec from <prefix>/libexec/gnat_ccg/bin. If not found, generate
   --  an error message on stdout and exit with status 1.

   -------------------------
   -- Executable_Location --
   -------------------------

   function Executable_Location return String is
      Exec_Name : constant String := Ada.Command_Line.Command_Name;

      function Get_Install_Dir (S : String) return String;
      --  S is the executable name preceeded by the absolute or relative path,
      --  e.g. "c:\usr\bin\gcc.exe" or "..\bin\gcc". Returns the absolute or
      --  relative directory where "bin" lies (in the example "C:\usr" or
      --  ".."). If the executable is not a "bin" directory, return "".

      ---------------------
      -- Get_Install_Dir --
      ---------------------

      function Get_Install_Dir (S : String) return String is
         Exec      : String  := GNAT.OS_Lib.Normalize_Pathname
           (S, Resolve_Links => True);
         Path_Last : Integer := 0;

      begin
         for J in reverse Exec'Range loop
            if Is_Directory_Separator (Exec (J)) then
               Path_Last := J - 1;
               exit;
            end if;
         end loop;

         if Path_Last >= Exec'First + 2 then
            GNAT.Case_Util.To_Lower (Exec (Path_Last - 2 .. Path_Last));
         end if;

         --  If we are not in a bin/ directory

         if Path_Last < Exec'First + 2
           or else Exec (Path_Last - 2 .. Path_Last) /= "bin"
           or else (Path_Last - 3 >= Exec'First
                    and then
                      not Is_Directory_Separator (Exec (Path_Last - 3)))
         then
            return Exec (Exec'First .. Path_Last)
              & GNAT.OS_Lib.Directory_Separator;

         else
            --  Skip bin/, but keep the last directory separator

            return Exec (Exec'First .. Path_Last - 3);
         end if;
      end Get_Install_Dir;

      --  Start of processing for Executable_Location

   begin
      --  First determine if a path prefix was placed in front of the
      --  executable name.

      for J in reverse Exec_Name'Range loop
         if Is_Directory_Separator (Exec_Name (J)) then
            return Get_Install_Dir (Exec_Name);
         end if;
      end loop;

      --  If you are here, the user has typed the executable name with no
      --  directory prefix.

      declare
         Ex  : String_Access   := GNAT.OS_Lib.Locate_Exec_On_Path (Exec_Name);
         Dir : constant String := Get_Install_Dir (Ex.all);

      begin
         Free (Ex);
         return Dir;
      end;
   end Executable_Location;

   ----------------------------
   -- Is_Directory_Separator --
   ----------------------------

   function Is_Directory_Separator (C : Character) return Boolean is
   begin
      --  In addition to the default directory_separator allow the '/' to act
      --  as separator.

      return C = Directory_Separator or else C = '/';
   end Is_Directory_Separator;

   type Host_Target is (X86_64_Linux, Aarch64_Linux);
   type Operation   is (Compile, Link, Other);

   HT         : Host_Target := x86_64_linux;
   Op         : Operation   := Link;

   GPU_Name   : String_Access;
   Ld         : Unbounded_String := To_Unbounded_String ("ld");
   Count      : constant Natural := Argument_Count;
   Path_Val   : constant String  := Value ("PATH", "");
   HT_Str     : String_Access := new String'("x86_64-linux");

   LLVM_Args  : Argument_List (1 .. Count + 1);
   --  We usually have to add -mcuda-libdevice= on top of the regular switches,
   --  so adding one to the input count.

   LLVM_Arg_Number : Integer := 0;

   Verbose   : Boolean := False;

   Prefix_LLVM_ARGS : constant Argument_List :=
     (new String'("--target=nvptx64"),
      new String'("-S"));

   Status    : Integer;

   Input_File_Number : Integer := 0;
   Input_Files : String_List (1 .. Argument_Count);
   Libdevice_Path : String_Access;

   function Spawn (S : String; Args : Argument_List) return Integer;
   --  Call GNAT.OS_Lib.Spawn and take Verbose into account

   function Locate_And_Check (Name : String) return String_Access;
   --  Locates a binary on path and raises an exception if not found.

   function Get_Argument
     (Arg : String; Prefix : String; Result : out Unbounded_String)
      return Boolean;
   --  If Prefix is found in Arg, then set the suffix in Result and returns
   --  true, else returns false.

   -----------
   -- Spawn --
   -----------

   function Spawn (S : String; Args : Argument_List) return Integer is
   begin
      if Verbose then
         Put (S);

         for J in Args'Range loop
            Put (" " & Args (J).all);
         end loop;

         New_Line;
      end if;

      return GNAT.OS_Lib.Spawn (S, Args);
   end Spawn;

   ----------------------
   -- Locate_And_Check --
   ----------------------

   function Locate_And_Check (Name : String) return String_Access is
      Tmp : String_Access;
   begin
      Tmp := Locate_Exec_On_Path (Name);

      if Tmp = null then
         Put_Line ("error: could not locate """ & Name & """ on PATH");
         raise Exec_Not_Found;
      end if;

      return Tmp;
   end Locate_And_Check;

   ------------------
   -- Get_Argument --
   ------------------

   function Get_Argument
     (Arg : String; Prefix : String; Result : out Unbounded_String) return Boolean is
   begin
      if Arg'Length >= Prefix'Length
        and then Arg (Arg'First .. Arg'First + Prefix'Length - 1) = Prefix
      then
         Result := To_Unbounded_String (Arg (Arg'First + Prefix'Length .. Arg'Last));
         return True;
      else
         return False;
      end if;
   end Get_Argument;

   --  Start of processing for GNATCCG_Wrapper

   CUDA_Root_Lazy : Unbounded_String := Null_Unbounded_String;
   --  Path to CUDA Root, resolved lazily when it is needed, not earlier
   function CUDA_Root return String is
   begin
      if CUDA_Root_Lazy = Null_Unbounded_String then
         declare
            CUDA_Bin : constant String := GNAT.Directory_Operations.Dir_Name
              (Locate_And_Check ("ptxas").all);
         begin
            Set_Unbounded_String
              (CUDA_Root_Lazy,
               (GNAT.Directory_Operations.Dir_Name
                    (CUDA_Bin (CUDA_Bin'First .. CUDA_Bin'Last - 1))));
         end;
      end if;

      return To_String (CUDA_Root_Lazy);
   end CUDA_Root;
begin
   for J in 1 .. Argument_Count loop
      declare
         Arg  : constant String := Argument (J);
         Sub_Arg : Unbounded_String;
      begin
         if Arg'Length > 0 and then Arg (1) /= '-'
           and then
             (J = 1 or else
               (Argument (J - 1) /= "-x"
               and then Argument (J - 1) /= "-o"))
         then
            --  Ignoring -o for now as we always generate file in the same
            --  format
            Input_File_Number := Input_File_Number + 1;

            if Arg (Arg'Last - 1 .. Arg'Last) = ".o" then
               --  There is a bug in gprlib that ignore gprconfig suffix for
               --  object files when it comes to binder files. As a temporary
               --  hack, changing the suffix here.
               Input_Files (Input_File_Number) :=
                 new String'(Arg (Arg'First .. Arg'Last - 2) & ".cubin");
            else
               Input_Files (Input_File_Number) :=
                 new String'(Arg);
            end if;

            LLVM_Arg_Number := @ + 1;
            LLVM_Args (LLVM_Arg_Number) := new String'(Arg);
         elsif Arg = "-c" or else Arg = "-S" then
            Op := Compile;
            LLVM_Arg_Number := @ + 1;
            LLVM_Args (LLVM_Arg_Number) := new String'(Arg);
         elsif Arg = "-v" or Arg = "--version" then
            Put_Line ("Target: cuda");
            Put_Line ("cuda-gcc version "
                      & gnatvsn.Library_Version
                      & " (for GNAT Pro "
                      & gnatvsn.Gnat_Static_Version_String
                      & ")");
            Put_Line ("CUDA Installation: " & CUDA_Root);

            Verbose := True;
            LLVM_Arg_Number := @ + 1;
            LLVM_Args (LLVM_Arg_Number) := new String'(Argument (J));
            if Argument_Count = 1 then
               GNAT.OS_Lib.OS_Exit (0);
            end if;
         elsif Get_Argument (Arg, "-mcpu=sm_", Sub_Arg) then
            GPU_Name := new String'(To_String (Sub_Arg));
         elsif Get_Argument (Arg, "-cuda-host=", Sub_Arg) then
            HT_Str := new String'(To_String (Sub_Arg));

            if Ht_Str.all = "aarch64-linux" then
               HT := Aarch64_Linux;
               LD := To_Unbounded_String ("aarch64-linux-gnu-ld");
            elsif Ht_Str.all = "x86_64-linux" then
               HT := X86_64_Linux;
               LD := To_Unbounded_String ("ld");
            else
               Put_Line ("-cuda-host " & Ht_Str.all & " not recognized. Supported: x86_64-linux (default), aarch64-linux.");
               Put_Line ("Proceeding with -cuda-host=x86_64-linux.");

               HT_Str := new String'("x86_64-linux");
            end if;
         elsif Get_Argument (Arg, "-mcuda-libdevice=", Sub_Arg) then
            Libdevice_Path := new String'(To_String (Sub_Arg));
         else
            LLVM_Arg_Number := @ + 1;
            LLVM_Args (LLVM_Arg_Number) := new String'(Argument (J));
         end if;
      end;
   end loop;

   if Libdevice_Path = null then
      --  We don't have a path for libdevice specified on the commande line,
      --  use heuristics to find the proper one

      declare
         Path_1 : String := CUDA_Root
           & "/lib/nvidia-cuda-toolkit/libdevice/libdevice.10.bc";
         Path_2 : String := CUDA_Root
           & "/nvvm/libdevice/libdevice.10.bc";
      begin
         if GNAT.IO_Aux.File_Exists (Path_1) then
            Libdevice_Path := new String'(Path_1);
         elsif GNAT.IO_Aux.File_Exists (Path_2) then
            Libdevice_Path := new String'(Path_2);
         else
            Put_Line
              ("error: could not locate libdevice on "
               & Path_1 & " or " & Path_2);

            return 1;
         end if;
      end;
   end if;

   if GPU_Name = null then
      GPU_Name := new String'("75");
   end if;

   case Op is
      when Compile =>
         LLVM_Arg_Number             := @ + 1;
         LLVM_Args (LLVM_Arg_Number) := new String'("-mcuda-libdevice=" & Libdevice_Path.all);

         if Input_File_Number /= 1 then
            Put ("error: expected one compilation file, got"
                     & Input_File_Number'Img & "(");

            for I in 1 .. Input_File_Number loop
               if I > 1 then
                  Put (", ");
               end if;

               Put (Input_Files (I).all);
            end loop;

            Put_Line (")");

            return 1;
         end if;

         declare
            File_Name : constant String :=
            Base_Name
               (Input_Files (1).all, File_Extension (Input_Files (1).all));
            PTX_Name : aliased String := File_Name & ".s";
            Obj_Name : aliased String := File_Name & ".cubin";

            PTXAS_Args : constant Argument_List :=
            (new String'("-arch=sm_" & GPU_Name.all),
               new String'("-m64"),
               new String'("--compile-only"),
               new String'("-v"),
               PTX_Name'Unchecked_Access,
               new String'("--output-file"),
               Obj_Name'Unchecked_Access);
         begin
            Status := Spawn
            (Locate_And_Check ("llvm-gcc").all,
               Prefix_LLVM_ARGS &
               new String'("-mcpu=sm_" & GPU_Name.all) &
               LLVM_Args (1 .. LLVM_Arg_Number));

            if Status /= 0 then
               Put_Line ("llvm-gcc failture");
               return Status;
            end if;

            Status := Spawn
            (Locate_And_Check ("ptxas").all,
               PTXAS_Args);

            if Status /= 0 then
               Put_Line ("ptxas failture");
               return Status;
            end if;
         end;
      when Link =>
         declare
            Kernel_Object : constant String := Input_Files (1).all;
            Kernel_Fat : constant String := Kernel_Object
            (Kernel_Object'First .. Kernel_Object'Last - 2);
            Kernel_Linked : constant String := Kernel_Object
            (Kernel_Object'First .. Kernel_Object'Last - 2) & ".linked";

            Nvlink_Args : constant Argument_List :=
            (new String'("--arch=sm_" & GPU_Name.all),
               new String'("-m64"),
               new String'("-L" & CUDA_Root & "targets/" & HT_Str.all & "/lib/stubs"),
               new String'("-L" & CUDA_Root & "targets/" & HT_Str.all & "/lib"))
            & Input_Files (2 .. Input_File_Number)
            &(new String'("-lcudadevrt"),
               new String'("-o"),
               new String'(Kernel_Linked));

            Fatbinary_Args : constant Argument_List :=
            (new String'("-64"),
               new String'("--create"),

               --  We currently only support one SAL per host application, with a
               --  fatbinary name hardcoded in gnatbind. We should eventually move
               --  to a scheme where we can have more than one. See U503-012
               --  new String'(Kernel_Fat),
               new String'("main.fatbin"),

               new String'("-link"),
               new String'("--image3=kind=elf,sm="
               & GPU_Name.all & ",file=" & Kernel_Linked));

            Ld_Args : constant Argument_List :=
            (new String'("-r"),
               new String'("-b"),
               new String'("binary"),
               new String'("main.fatbin"),
               new String'("-o"),
               new String'(Kernel_Object)
            );
         begin
            Status := Spawn
            (Locate_And_Check ("nvlink").all,
               Nvlink_Args);

            if Status /= 0 then
               Put_Line ("nvlink failure");
               return Status;
            end if;

            Status := Spawn
            (Locate_And_Check ("fatbinary").all,
               Fatbinary_Args);

            if Status /= 0 then
               Put_Line ("fatbinary failure");
               return Status;
            end if;

            Status := Spawn (Locate_And_Check (To_String (Ld)).all, Ld_Args);

            if Status /= 0 then
               Put_Line ("ld failure");
               return Status;
            end if;
         end;
      when Other => null;
   end case;

   return 0;
exception
   when Exec_Not_Found =>
      return 1;
end GNATCUDA_Wrapper;
