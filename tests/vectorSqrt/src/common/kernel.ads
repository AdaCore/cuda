with System;

package Kernel is

   type Float_Array is array (Integer range <>) of Float;

   type Access_Host_Float_Array is access all Float_Array;

   procedure Vector_Sqrt
     (A_Addr : System.Address;
      B_Addr : System.Address;
      Num_Elements : Integer)
     with CUDA_Global;

end Kernel;
