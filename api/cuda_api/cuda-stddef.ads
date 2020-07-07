with Interfaces.C; use Interfaces.C;

package CUDA.Stddef is
   type Max_Align_T is record
      Uu_Max_Align_Ll : Long_Long_Integer;
      Uu_Max_Align_Ld : long_double;
   end record;
end CUDA.Stddef;
