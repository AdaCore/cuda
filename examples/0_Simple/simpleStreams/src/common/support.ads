with CUDA.Storage_Models; use CUDA.Storage_Models;

package Support is

   Stream_Model : CUDA_Async_Storage_Model;

   type Integer_Array is array (Natural range <>) of Integer;

   type Integer_Array_Host_Access is access all Integer_Array;

   type Integer_Array_Device_Access is access Integer_Array
     with Designated_Storage_Model => Stream_Model;

end Support;
