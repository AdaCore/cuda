pragma Ada_2012;
pragma Style_Checks (Off);
pragma Warnings ("U");

with Interfaces.C; use Interfaces.C;
with System;

package vadefs_h is

   subtype uu_gnuc_va_list is System.Address;  -- c:\home\ochem\sandbox\x86_64-windows\gnat\install\x86_64-pc-mingw32\include\vadefs.h:24

   subtype va_list is uu_gnuc_va_list;  -- c:\home\ochem\sandbox\x86_64-windows\gnat\install\x86_64-pc-mingw32\include\vadefs.h:31

end vadefs_h;
