with System;
with Ada.Unchecked_Deallocation;
with CUDA.Runtime_Api;  use CUDA.Runtime_Api;

package Kernel is

   type Float_Array is array (Integer range <>) of Float;

   type Access_Host_Float_Array is access all Float_Array;

   procedure Free is new
     Ada.Unchecked_Deallocation (Float_Array, Access_Host_Float_Array);

   procedure Vector_Add
     (A_Addr : System.Address;
      B_Addr : System.Address;
      C_Addr : System.Address;
      Num_Elements : Integer)
     with CUDA_Global;

   --  procedure Initialize_Cuda_Kernel;

   --  pragma Linker_Constructor (Initialize_Cuda_Kernel);

end Kernel;
