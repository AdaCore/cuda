with System;

with Storage_Models;

package CUDA_Storage_Models is

   function Malloc_Allocate (Size : Natural) return System.Address with Inline;
   procedure Malloc_Deallocate (Address : System.Address) with Inline;
   procedure Malloc_Copy_To_Foreign (Dst : System.Address; Src : System.Address; Bytes : Natural) with Inline;
   procedure Malloc_Copy_To_Native (Dst : System.Address; Src : System.Address; Bytes : Natural) with Inline;

   package Malloc_Storage_Model is new Storage_Models
     (Foreign_Address => System.Address,
      Allocate        => Malloc_Allocate,
      Deallocate      => Malloc_Deallocate,
      Copy_To_Foreign => Malloc_Copy_To_Foreign,
      Copy_To_Native  => Malloc_Copy_To_Native);

   function Malloc_Host_Allocate (Size : Natural) return System.Address with Inline;
   procedure Malloc_Host_Deallocate (Address : System.Address) with Inline;
   procedure Malloc_Host_Copy_To_Foreign (Dst : System.Address; Src : System.Address; Bytes : Natural) with Inline;
   procedure Malloc_Host_Copy_To_Native (Dst : System.Address; Src : System.Address; Bytes : Natural) with Inline;

   package Malloc_Host_Storage_Model is new Storage_Models
     (Foreign_Address => System.Address,
      Allocate        => Malloc_Host_Allocate,
      Deallocate      => Malloc_Host_Deallocate,
      Copy_To_Foreign => Malloc_Host_Copy_To_Foreign,
      Copy_To_Native  => Malloc_Host_Copy_To_Native);

end CUDA_Storage_Models;
