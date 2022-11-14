with CUDA.Runtime_Api; use CUDA.Runtime_Api;
with Interfaces.C;     use Interfaces.C;

package body Kernel is

   procedure Complex_Computation
     (A : Float_Array; B : Float_Array; C : out Float_Array; I : Integer)
   is
   begin
      if I < A'Length then
         for J in A'First + I .. A'Last loop
            for K in B'First + I .. B'Last loop
               C (C'First + I) := A (J) + B (K);
            end loop;
         end loop;
      end if;
   end Complex_Computation;

end Kernel;
