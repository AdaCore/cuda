with System; use System;
with System.Storage_Elements; use System.Storage_Elements;

with CUDA.Driver_Types; use CUDA.Driver_Types;

package CUDA.Storage_Model is

   type CUDA_Address is new System.Address;

   type CUDA_Storage_Model is limited record
      Stream : CUDA.Driver_Types.Stream_T := null;
      Async  : Boolean := False;
   end record
     with Storage_Model_Type =>
       (Address_Type          => CUDA_Address,
        Allocate              => CUDA_Allocate,
        Deallocate            => CUDA_Deallocate,
        Copy_To               => CUDA_Copy_To,
        Copy_From             => CUDA_Copy_From,
        Storage_Size          => CUDA_Storage_Size,
        Null_Address          => CUDA_Null_Address);

   CUDA_Null_Address : constant CUDA_Address :=
     CUDA_Address (System.Null_Address);

   procedure CUDA_Allocate
     (Model           : in out CUDA_Storage_Model;
      Storage_Address : out CUDA_Address;
      Size            : Storage_Count;
      Alignment       : Storage_Count);

   procedure CUDA_Deallocate
     (Model           : in out CUDA_Storage_Model;
      Storage_Address : CUDA_Address;
      Size            : Storage_Count;
      Alignment       : Storage_Count);

   procedure CUDA_Copy_To
     (Model  : in out CUDA_Storage_Model;
      Target : CUDA_Address;
      Source : System.Address;
      Size   : Storage_Count);

   procedure CUDA_Copy_From
     (Model  : in out CUDA_Storage_Model;
      Target : System.Address;
      Source : CUDA_Address;
      Size   : Storage_Count);

   function CUDA_Storage_Size
     (Model : CUDA_Storage_Model)
   return Storage_Count;

   Model : CUDA_Storage_Model;

end CUDA.Storage_Model;
