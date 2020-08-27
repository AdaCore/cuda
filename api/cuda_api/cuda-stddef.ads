with Interfaces.C; use Interfaces.C;

package CUDA.Stddef is

   subtype Ptrdiff_T is long;
   subtype Size_T is unsigned_long;
   subtype Wchar_T is int;
   type Max_Align_T is record
      Uu_Max_Align_Ll : Long_Long_Integer;
      Uu_Max_Align_Ld : long_double;

   end record with
      Convention => C;

end CUDA.Stddef;
