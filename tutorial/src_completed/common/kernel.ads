with System;

with CUDA.Storage_Models; use CUDA.Storage_Models;

package Kernel is

   type Float_Array is array (Integer range <>) of Float;

   type Array_Device_Access is access Float_Array
     with Designated_Storage_Model => CUDA.Storage_Models.Model;

    procedure Native_Complex_Computation
     (A : Float_Array;
      B : Float_Array;
      C : out Float_Array;
      I : Integer);

   procedure Device_Complex_Computation
     (A : Array_Device_Access;
      B : Array_Device_Access;
      C : Array_Device_Access)
     with CUDA_Global;

end Kernel;
