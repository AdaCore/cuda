with System;

generic
   type Foreign_Address is private;
   type Copy_Options is private;
   Default_Copy_Options : Copy_Options;

   with function Allocate (Size : Natural) return Foreign_Address;
   with procedure Deallocate (Address : Foreign_Address);
   with procedure Copy_To_Foreign (Dst : Foreign_Address; Src : System.Address; Bytes : Natural; Options : Copy_Options);
   with procedure Copy_To_Native (Dst : System.Address; Src : Foreign_Address; Bytes : Natural; Options : Copy_Options);
   with function Offset (Address : Foreign_Address; Bytes : Natural) return Foreign_Address;
package Storage_Models is

end Storage_Models;
