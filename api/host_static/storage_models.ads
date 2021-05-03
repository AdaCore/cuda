with System;

generic
   type Foreign_Address is private;

   with function Allocate (Size : Natural) return Foreign_Address;
   with procedure Deallocate (Address : Foreign_Address);
   with procedure Copy_To_Foreign (Dst : Foreign_Address; Dst_Offset : Natural; Src : System.Address; Bytes : Natural);
   with procedure Copy_To_Native (Dst : System.Address; Src : Foreign_Address; Src_Offset : Natural; Bytes : Natural);
package Storage_Models is

end Storage_Models;
