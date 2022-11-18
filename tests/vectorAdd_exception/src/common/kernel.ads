with System;

with CUDA.Storage_Models; use CUDA.Storage_Models;

package Kernel is

   type Float_Array is array (Integer range <>) of Float;

   type Array_Device_Access is access Float_Array
     with Designated_Storage_Model => CUDA.Storage_Models.Model;

   procedure Vector_Add
     (A : Array_Device_Access;
      B : Array_Device_Access;
      C : Array_Device_Access)
     with CUDA_Global;

  procedure Last_Chance_Handler is null;
  pragma Export (C, Last_Chance_Handler, "__gnat_last_chance_handler");

end Kernel;
