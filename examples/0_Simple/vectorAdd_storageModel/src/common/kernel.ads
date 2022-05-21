with System;

package Kernel is

   type Float_Array is array (Integer range <>) of Float;

   type Access_Host_Float_Array is access all Float_Array;

   procedure Vector_Add
     (A : Access_Host_Float_Array;
      B : Access_Host_Float_Array;
      C : Access_Host_Float_Array)
     with CUDA_Global;

end Kernel;
