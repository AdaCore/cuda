with CUDA.Storage_Models;

package Kernels is

   type Int_Array is array (Integer range 0 .. <>) of Integer;

   Stream_Model : CUDA.Storage_Models.CUDA_Async_Storage_Model;

   type Array_Device_Access is access Int_Array with
     Designated_Storage_Model => Stream_Model;

   procedure Increment_Kernel
     (G_Data : Array_Device_Access; Inc_Value : Integer) with
     Cuda_Global;

end Kernels;
