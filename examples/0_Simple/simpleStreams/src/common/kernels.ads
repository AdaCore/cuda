with Support; use Support;

package Kernels is

   procedure Init_Array
     (G_Data : Integer_Array_Access;
      Factor : Integer_Array_Access;
      Num_Iterations : Integer) with CUDA_Global;

end Kernels;
