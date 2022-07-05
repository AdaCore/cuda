with System; use System;
with System.Storage_Elements; use System.Storage_Elements;

with CUDA.Driver_Types; use CUDA.Driver_Types;

package CUDA.Storage_Models is

   type CUDA_Address is new System.Address;

   type CUDA_Storage_Model is limited record
      null;
   end record
     with Storage_Model_Type;

   type CUDA_Async_Storage_Model is limited record
      null;
   end record
     with Storage_Model_Type;

   Model : CUDA_Storage_Model;

end CUDA.Storage_Models;
