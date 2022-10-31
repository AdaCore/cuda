with CUDA.Runtime_Api; use CUDA.Runtime_Api;
with Interfaces.C;     use Interfaces.C;

package body Kernel is

    procedure Native_Complex_Computation
     (A : Float_Array;
      B : Float_Array;
      C : out Float_Array;
      I : Integer)
   is
   begin
      if I < A'Length then
         for J in A'First + I .. A'Last loop
            for K in B'First + I .. B'Last loop
               C (C'First + I) := A (J) + B (K);
            end loop;
         end loop;
      end if;
   end Native_Complex_Computation;

   procedure Device_Complex_Computation
     (A : Array_Device_Access;
      B : Array_Device_Access;
      C : Array_Device_Access)
   is
      I : Integer := Integer (Block_Dim.X * Block_IDx.X + Thread_IDx.X);
   begin
      Native_Complex_Computation (A.all, B.all, C.all, I);
   end Device_Complex_Computation;

end Kernel;
