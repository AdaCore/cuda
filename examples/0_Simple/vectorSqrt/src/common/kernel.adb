with CUDA.Runtime_Api; use CUDA.Runtime_Api; -- Block_Dim, Block_IDx, Thread_IDx
with Interfaces.C;     use Interfaces.C; -- Operators for Block_Dim, Block_IDx, Thread_IDx

with Ada.Numerics; use Ada.Numerics;
with Ada.Numerics.Generic_Elementary_Functions;

package body Kernel is

   procedure Vector_Sqrt
     (A_Addr : System.Address;
      B_Addr : System.Address;
      Num_Elements : Integer)
   is
      A : Float_Array (1..Num_Elements) with Address => A_Addr;
      B : Float_Array (1..Num_Elements) with Address => B_Addr;
      I : Integer := Integer (Block_Dim.X * Block_IDx.X + Thread_IDx.X);
      package Elementary_Functions is new
         Ada.Numerics.Generic_Elementary_Functions (Float);
   begin
      if I < Num_Elements then
         B (B'First + I) := Elementary_Functions.Sqrt (A (A'First + I));
      end if;
   end Vector_Sqrt;

end Kernel;
