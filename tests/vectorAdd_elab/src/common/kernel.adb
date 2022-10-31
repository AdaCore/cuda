with CUDA.Runtime_Api; use CUDA.Runtime_Api; -- Block_Dim, Block_IDx, Thread_IDx
with Interfaces.C;     use Interfaces.C; -- Operators for Block_Dim, Block_IDx, Thread_IDx

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
         C (C'First + I) := A (A'First + I) + B (B'First + I) + Elaborated_Value;
      end if;
   end Vector_Add;

begin

   for I in 1 ..  10_000 loop
      Elaborated_Value := Elaborated_Value + 1.0;
   end loop;

end Kernel;
