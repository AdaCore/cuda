with Interfaces.C; use Interfaces.C;
with Interfaces.C.Extensions;
with Interfaces.C.Strings;

package CUDA.Corecrt is
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

   type Pthreadlocinfo is access Threadlocaleinfostruct;

   type Pthreadmbcinfo is access Threadmbcinfostruct;

   type Uu_Lc_Time_Data is null record;

   type Localeinfo_Struct is record
      Locinfo : Pthreadlocinfo;
      Mbcinfo : Pthreadmbcinfo;
   end record;
   subtype U_Locale_Tstruct is Localeinfo_Struct;

   type U_Locale_T is access Localeinfo_Struct;

   type Tag_LC_ID is record
      W_Language  : unsigned_short;
      W_Country   : unsigned_short;
      W_Code_Page : unsigned_short;
   end record;
   subtype LC_ID is Tag_LC_ID;

   type LPLC_ID is access Tag_LC_ID;

   type Lconv is null record;

   type Anon919_Array935 is array (0 .. 5) of unsigned_long;

   type Anon919_Array936 is array (0 .. 5) of LC_ID;

   type Anon919_Struct938 is access Interfaces.C.Strings.chars_ptr;

   type Anon919_Array941 is array (0 .. 5) of Anon919_Struct938;

   type Threadlocaleinfostruct is access int;
   subtype Threadlocinfo is Threadlocaleinfostruct;
end CUDA.Corecrt;
