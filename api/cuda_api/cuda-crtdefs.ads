with Interfaces.C;
with Interfaces.C.Extensions;
with Interfaces.C.Strings; use Interfaces.C;

package CUDA.Crtdefs is
   subtype Size_T is Extensions.unsigned_long_long;
   subtype Ssize_T is Long_Long_Integer;
   subtype Rsize_T is Size_T;
   subtype Intptr_T is Long_Long_Integer;
   subtype Uintptr_T is Extensions.unsigned_long_long;
   subtype Ptrdiff_T is Long_Long_Integer;
   subtype Wchar_T is unsigned_short;
   subtype Wint_T is unsigned_short;
   subtype Wctype_T is unsigned_short;
   subtype Errno_T is int;
   subtype Uu_Time32_T is long;
   subtype Uu_Time64_T is Long_Long_Integer;
   subtype Time_T is Uu_Time64_T;

   type Threadmbcinfostruct is null record;
   type Threadlocaleinfostruct;

   type Pthreadlocinfo is access all Threadlocaleinfostruct;

   type Pthreadmbcinfo is access all Threadmbcinfostruct;

   type Uu_Lc_Time_Data is null record;

   type Localeinfo_Struct is record
      Locinfo : Pthreadlocinfo;
      Mbcinfo : Pthreadmbcinfo;
   end record with
      Convention => C_Pass_By_Copy;
   subtype U_Locale_Tstruct is Localeinfo_Struct;

   type U_Locale_T is access all Localeinfo_Struct;

   type Tag_LC_ID is record
      W_Language  : aliased unsigned_short;
      W_Country   : aliased unsigned_short;
      W_Code_Page : aliased unsigned_short;
   end record with
      Convention => C_Pass_By_Copy;
   subtype LC_ID is Tag_LC_ID;

   type LPLC_ID is access all Tag_LC_ID;

   type Lconv is null record;

   type Anon919_Lc_Handle_Array is array (0 .. 5) of aliased unsigned_long;

   type Anon919_Lc_Id_Array is array (0 .. 5) of aliased LC_ID;

   type Anon919_Anon938_Struct is record
      Locale    : Interfaces.C.Strings.chars_ptr;
      Wlocale   : access Wchar_T;
      Refcount  : access int;
      Wrefcount : access int;
   end record with
      Convention => C_Pass_By_Copy;

   type Anon919_Lc_Category_Array is array (0 .. 5) of aliased Anon919_Anon938_Struct;

   type Threadlocaleinfostruct is record
      Refcount            : aliased int;
      Lc_Codepage         : aliased unsigned;
      Lc_Collate_Cp       : aliased unsigned;
      Lc_Handle           : aliased Anon919_Lc_Handle_Array;
      Lc_Id               : aliased Anon919_Lc_Id_Array;
      Lc_Category         : aliased Anon919_Lc_Category_Array;
      Lc_Clike            : aliased int;
      Mb_Cur_Max          : aliased int;
      Lconv_Intl_Refcount : access int;
      Lconv_Num_Refcount  : access int;
      Lconv_Mon_Refcount  : access int;
      The_Lconv           : access Lconv;
      Ctype1_Refcount     : access int;
      Ctype1              : access unsigned_short;
      Pctype              : access unsigned_short;
      Pclmap              : access unsigned_char;
      Pcumap              : access unsigned_char;
      Lc_Time_Curr        : access Uu_Lc_Time_Data;
   end record with
      Convention => C_Pass_By_Copy;
   subtype Threadlocinfo is Threadlocaleinfostruct;
end CUDA.Crtdefs;
