with Ada.Unchecked_Deallocation;

with CUDA.Runtime_Api;  use CUDA.Runtime_Api;

package Kernel is

   type Float_Array is array (Integer range <>) of Float;

   type Access_Device_Float_Array is access all Float_Array
     with Storage_Pool => CUDA_Device;

   type Access_Host_Float_Array is access all Float_Array;

   procedure Free is new
     Ada.Unchecked_Deallocation (Float_Array, Access_Device_Float_Array);

   procedure Free is new
     Ada.Unchecked_Deallocation (Float_Array, Access_Host_Float_Array);

   procedure Vector_Add
     (A : Float_Array;
      B : Float_Array;
      C : out Float_Array;
      Num_Elements : Integer)
     with CUDA_Global;

end Kernel;
