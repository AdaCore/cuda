with Interfaces.C; use Interfaces.C;
with CUDA.Runtime_Api; use CUDA.Runtime_Api;

package body Kernels is

   procedure Init_Array
     (G_Data : Integer_Array_Device_Access;
      From, To : Integer;
      Factor : Integer_Array_Device_Access;
      Num_Iterations : Integer)
   is
      Idx : Integer := Integer (Block_Idx.X * Block_Dim.X + Thread_Idx.X);
   begin
      for I in 1 .. Num_Iterations loop
         G_Data (From) := @ + Factor (Factor'First);
      end loop;
   end Init_Array;


end Kernels;
