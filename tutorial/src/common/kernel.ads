with System;

with CUDA.Storage_Models; use CUDA.Storage_Models;

package Kernel is

   type Float_Array is array (Integer range <>) of Float;

   procedure Native_Complex_Computation
     (A : Float_Array;
      B : Float_Array;
      C : out Float_Array;
      I : Integer);

end Kernel;
