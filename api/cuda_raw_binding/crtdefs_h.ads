pragma Ada_2012;
pragma Style_Checks (Off);

with Interfaces.C; use Interfaces.C;
with Interfaces.C.Extensions;
with Interfaces.C.Strings;

package crtdefs_h is

   subtype size_t is Extensions.unsigned_long_long;  -- c:\home\ochem\sandbox\x86_64-windows\gnat\install\x86_64-pc-mingw32\include\crtdefs.h:35

   subtype ssize_t is Long_Long_Integer;  -- c:\home\ochem\sandbox\x86_64-windows\gnat\install\x86_64-pc-mingw32\include\crtdefs.h:45

   subtype rsize_t is size_t;  -- c:\home\ochem\sandbox\x86_64-windows\gnat\install\x86_64-pc-mingw32\include\crtdefs.h:52

   subtype intptr_t is Long_Long_Integer;  -- c:\home\ochem\sandbox\x86_64-windows\gnat\install\x86_64-pc-mingw32\include\crtdefs.h:62

   subtype uintptr_t is Extensions.unsigned_long_long;  -- c:\home\ochem\sandbox\x86_64-windows\gnat\install\x86_64-pc-mingw32\include\crtdefs.h:75

   subtype ptrdiff_t is Long_Long_Integer;  -- c:\home\ochem\sandbox\x86_64-windows\gnat\install\x86_64-pc-mingw32\include\crtdefs.h:88

   subtype wchar_t is unsigned_short;  -- c:\home\ochem\sandbox\x86_64-windows\gnat\install\x86_64-pc-mingw32\include\crtdefs.h:98

   subtype wint_t is unsigned_short;  -- c:\home\ochem\sandbox\x86_64-windows\gnat\install\x86_64-pc-mingw32\include\crtdefs.h:106

   subtype wctype_t is unsigned_short;  -- c:\home\ochem\sandbox\x86_64-windows\gnat\install\x86_64-pc-mingw32\include\crtdefs.h:107

   subtype errno_t is int;  -- c:\home\ochem\sandbox\x86_64-windows\gnat\install\x86_64-pc-mingw32\include\crtdefs.h:113

   subtype uu_time32_t is long;  -- c:\home\ochem\sandbox\x86_64-windows\gnat\install\x86_64-pc-mingw32\include\crtdefs.h:118

   subtype uu_time64_t is Long_Long_Integer;  -- c:\home\ochem\sandbox\x86_64-windows\gnat\install\x86_64-pc-mingw32\include\crtdefs.h:123

   subtype time_t is uu_time64_t;  -- c:\home\ochem\sandbox\x86_64-windows\gnat\install\x86_64-pc-mingw32\include\crtdefs.h:138

   type threadmbcinfostruct is null record;   -- incomplete struct

   type threadlocaleinfostruct;
   type pthreadlocinfo is access all threadlocaleinfostruct;  -- c:\home\ochem\sandbox\x86_64-windows\gnat\install\x86_64-pc-mingw32\include\crtdefs.h:424

   type pthreadmbcinfo is access all threadmbcinfostruct;  -- c:\home\ochem\sandbox\x86_64-windows\gnat\install\x86_64-pc-mingw32\include\crtdefs.h:425

   type uu_lc_time_data is null record;   -- incomplete struct

   type localeinfo_struct is record
      locinfo : pthreadlocinfo;  -- c:\home\ochem\sandbox\x86_64-windows\gnat\install\x86_64-pc-mingw32\include\crtdefs.h:429
      mbcinfo : pthreadmbcinfo;  -- c:\home\ochem\sandbox\x86_64-windows\gnat\install\x86_64-pc-mingw32\include\crtdefs.h:430
   end record
   with Convention => C_Pass_By_Copy;  -- c:\home\ochem\sandbox\x86_64-windows\gnat\install\x86_64-pc-mingw32\include\crtdefs.h:428

   subtype u_locale_tstruct is localeinfo_struct;  -- c:\home\ochem\sandbox\x86_64-windows\gnat\install\x86_64-pc-mingw32\include\crtdefs.h:431

   type u_locale_t is access all localeinfo_struct;  -- c:\home\ochem\sandbox\x86_64-windows\gnat\install\x86_64-pc-mingw32\include\crtdefs.h:431

   type tagLC_ID is record
      wLanguage : aliased unsigned_short;  -- c:\home\ochem\sandbox\x86_64-windows\gnat\install\x86_64-pc-mingw32\include\crtdefs.h:436
      wCountry : aliased unsigned_short;  -- c:\home\ochem\sandbox\x86_64-windows\gnat\install\x86_64-pc-mingw32\include\crtdefs.h:437
      wCodePage : aliased unsigned_short;  -- c:\home\ochem\sandbox\x86_64-windows\gnat\install\x86_64-pc-mingw32\include\crtdefs.h:438
   end record
   with Convention => C_Pass_By_Copy;  -- c:\home\ochem\sandbox\x86_64-windows\gnat\install\x86_64-pc-mingw32\include\crtdefs.h:435

   subtype LC_ID is tagLC_ID;  -- c:\home\ochem\sandbox\x86_64-windows\gnat\install\x86_64-pc-mingw32\include\crtdefs.h:439

   type LPLC_ID is access all tagLC_ID;  -- c:\home\ochem\sandbox\x86_64-windows\gnat\install\x86_64-pc-mingw32\include\crtdefs.h:439

   type lconv is null record;   -- incomplete struct

   type anon919_lc_handle_array is array (0 .. 5) of aliased unsigned_long;
   type anon919_lc_id_array is array (0 .. 5) of aliased LC_ID;
   type anon919_anon938_struct is record
      locale : Interfaces.C.Strings.chars_ptr;  -- c:\home\ochem\sandbox\x86_64-windows\gnat\install\x86_64-pc-mingw32\include\crtdefs.h:451
      wlocale : access wchar_t;  -- c:\home\ochem\sandbox\x86_64-windows\gnat\install\x86_64-pc-mingw32\include\crtdefs.h:452
      refcount : access int;  -- c:\home\ochem\sandbox\x86_64-windows\gnat\install\x86_64-pc-mingw32\include\crtdefs.h:453
      wrefcount : access int;  -- c:\home\ochem\sandbox\x86_64-windows\gnat\install\x86_64-pc-mingw32\include\crtdefs.h:454
   end record
   with Convention => C_Pass_By_Copy;
   type anon919_lc_category_array is array (0 .. 5) of aliased anon919_anon938_struct;
   type threadlocaleinfostruct is record
      refcount : aliased int;  -- c:\home\ochem\sandbox\x86_64-windows\gnat\install\x86_64-pc-mingw32\include\crtdefs.h:445
      lc_codepage : aliased unsigned;  -- c:\home\ochem\sandbox\x86_64-windows\gnat\install\x86_64-pc-mingw32\include\crtdefs.h:446
      lc_collate_cp : aliased unsigned;  -- c:\home\ochem\sandbox\x86_64-windows\gnat\install\x86_64-pc-mingw32\include\crtdefs.h:447
      lc_handle : aliased anon919_lc_handle_array;  -- c:\home\ochem\sandbox\x86_64-windows\gnat\install\x86_64-pc-mingw32\include\crtdefs.h:448
      lc_id : aliased anon919_lc_id_array;  -- c:\home\ochem\sandbox\x86_64-windows\gnat\install\x86_64-pc-mingw32\include\crtdefs.h:449
      lc_category : aliased anon919_lc_category_array;  -- c:\home\ochem\sandbox\x86_64-windows\gnat\install\x86_64-pc-mingw32\include\crtdefs.h:455
      lc_clike : aliased int;  -- c:\home\ochem\sandbox\x86_64-windows\gnat\install\x86_64-pc-mingw32\include\crtdefs.h:456
      mb_cur_max : aliased int;  -- c:\home\ochem\sandbox\x86_64-windows\gnat\install\x86_64-pc-mingw32\include\crtdefs.h:457
      lconv_intl_refcount : access int;  -- c:\home\ochem\sandbox\x86_64-windows\gnat\install\x86_64-pc-mingw32\include\crtdefs.h:458
      lconv_num_refcount : access int;  -- c:\home\ochem\sandbox\x86_64-windows\gnat\install\x86_64-pc-mingw32\include\crtdefs.h:459
      lconv_mon_refcount : access int;  -- c:\home\ochem\sandbox\x86_64-windows\gnat\install\x86_64-pc-mingw32\include\crtdefs.h:460
      the_lconv : access lconv;  -- c:\home\ochem\sandbox\x86_64-windows\gnat\install\x86_64-pc-mingw32\include\crtdefs.h:461
      ctype1_refcount : access int;  -- c:\home\ochem\sandbox\x86_64-windows\gnat\install\x86_64-pc-mingw32\include\crtdefs.h:462
      ctype1 : access unsigned_short;  -- c:\home\ochem\sandbox\x86_64-windows\gnat\install\x86_64-pc-mingw32\include\crtdefs.h:463
      pctype : access unsigned_short;  -- c:\home\ochem\sandbox\x86_64-windows\gnat\install\x86_64-pc-mingw32\include\crtdefs.h:464
      pclmap : access unsigned_char;  -- c:\home\ochem\sandbox\x86_64-windows\gnat\install\x86_64-pc-mingw32\include\crtdefs.h:465
      pcumap : access unsigned_char;  -- c:\home\ochem\sandbox\x86_64-windows\gnat\install\x86_64-pc-mingw32\include\crtdefs.h:466
      lc_time_curr : access uu_lc_time_data;  -- c:\home\ochem\sandbox\x86_64-windows\gnat\install\x86_64-pc-mingw32\include\crtdefs.h:467
   end record
   with Convention => C_Pass_By_Copy;  -- c:\home\ochem\sandbox\x86_64-windows\gnat\install\x86_64-pc-mingw32\include\crtdefs.h:444

   subtype threadlocinfo is threadlocaleinfostruct;  -- c:\home\ochem\sandbox\x86_64-windows\gnat\install\x86_64-pc-mingw32\include\crtdefs.h:468

end crtdefs_h;
