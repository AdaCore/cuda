with System;

with Storage_Models;
with CUDA.Driver_Types;

package CUDA_Storage_Models is

   type Copy_Options (Async : Boolean := False) is record
      Stream : CUDA.Driver_Types.Stream_T;
   end record;

   Default_Copy_Options : Copy_Options (False);

   function Address_Offset (Address : System.Address; Bytes : Natural) return System.Address;

   function Malloc_Allocate (Size : Natural) return System.Address with Inline;
   procedure Malloc_Deallocate (Address : System.Address) with Inline;
   procedure Malloc_Copy_To_Foreign (Dst : System.Address; Src : System.Address; Bytes : Natural; Options : Copy_Options) with Inline;
   procedure Malloc_Copy_To_Native (Dst : System.Address; Src : System.Address; Bytes : Natural; Options : Copy_Options) with Inline;

   package Malloc_Storage_Model is new Storage_Models
     (Foreign_Address      => System.Address,
      Copy_Options         => Copy_Options,
      Default_Copy_Options => Default_Copy_Options,
      Allocate             => Malloc_Allocate,
      Deallocate           => Malloc_Deallocate,
      Copy_To_Foreign      => Malloc_Copy_To_Foreign,
      Copy_To_Native       => Malloc_Copy_To_Native,
      Offset               => Address_Offset);

   function Malloc_Host_Allocate (Size : Natural) return System.Address with Inline;
   procedure Malloc_Host_Deallocate (Address : System.Address) with Inline;
   procedure Malloc_Host_Copy_To_Foreign (Dst : System.Address; Src : System.Address; Bytes : Natural; Options : Copy_Options) with Inline;
   procedure Malloc_Host_Copy_To_Native (Dst : System.Address; Src : System.Address; Bytes : Natural; Options : Copy_Options) with Inline;

   package Malloc_Host_Storage_Model is new Storage_Models
     (Foreign_Address      => System.Address,
      Copy_Options         => Copy_Options,
      Default_Copy_Options => Default_Copy_Options,
      Allocate             => Malloc_Host_Allocate,
      Deallocate           => Malloc_Host_Deallocate,
      Copy_To_Foreign      => Malloc_Host_Copy_To_Foreign,
      Copy_To_Native       => Malloc_Host_Copy_To_Native,
      Offset               => Address_Offset);

end CUDA_Storage_Models;
