with Interfaces.C; use Interfaces.C;

package CUDA.Stddef is
   type Max_Align_T is record
      Uu_Max_Align_Ll : aliased Long_Long_Integer;
      Uu_Max_Align_Ld : aliased long_double;
   end record with
      Convention => C_Pass_By_Copy;
end CUDA.Stddef;
