with Interfaces.C; use Interfaces.C;

with CUDA.Runtime_Api;  use CUDA.Runtime_Api;

package body Kernel is
   
   procedure Vector_Add
     (A : Float_Array;
      B : Float_Array;
      C : out Float_Array;
      Num_Elements : Integer)
   is
      --  I : Integer := Integer (Block_Dim.X * Block_IDx.X + Thread_IDx.X);
   begin
      --  if I < Num_Elements then
      --     C (C'First + I) := A (A'First + I) + B (B'First + I);
      --  end if;
      null;
   end Vector_Add;
   
end Kernel;
