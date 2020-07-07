pragma Ada_2012;
pragma Style_Checks (Off);
pragma Warnings ("U");

with Interfaces.C; use Interfaces.C;

package stddef_h is

   --  unsupported macro: NULL ((void *)0)
   --  arg-macro: procedure offsetof (TYPE, MEMBER)
   --    __builtin_offsetof (TYPE, MEMBER)
   type max_align_t is record
      uu_max_align_ll : aliased Long_Long_Integer;  -- c:\home\ochem\sandbox\x86_64-windows\gnat\install\lib\gcc\x86_64-pc-mingw32\9.3.1\include\stddef.h:416
      uu_max_align_ld : aliased long_double;  -- c:\home\ochem\sandbox\x86_64-windows\gnat\install\lib\gcc\x86_64-pc-mingw32\9.3.1\include\stddef.h:417
   end record
   with Convention => C_Pass_By_Copy;  -- c:\home\ochem\sandbox\x86_64-windows\gnat\install\lib\gcc\x86_64-pc-mingw32\9.3.1\include\stddef.h:426

end stddef_h;
