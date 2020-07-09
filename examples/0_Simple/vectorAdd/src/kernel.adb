with Interfaces.C; use Interfaces.C;
with Interfaces.C.Strings;
with CUDA.GPU_Api;  use CUDA.GPU_Api;
with Interfaces; use Interfaces;
with Interfaces.C;            use Interfaces.C;
with Interfaces.C.Extensions; use Interfaces.C.Extensions;
with Interfaces.C.Strings;

package body Kernel is
   
   procedure Vector_Add
     (A_Addr : System.Address;
      B_Addr : System.Address;
      C_Addr : System.Address;
      Num_Elements : Integer)
   is
      A : Float_Array (1..Num_Elements) with Address => A_Addr;
      B : Float_Array (1..Num_Elements) with Address => B_Addr;
      C : Float_Array (1..Num_Elements) with Address => C_Addr;
      I : Integer := Integer (Block_Dim.X * Block_IDx.X + Thread_IDx.X);
   begin
      if I < Num_Elements then
         C (C'First + I) := A (A'First + I) + B (B'First + I);
      end if;
   end Vector_Add;

end Kernel;
