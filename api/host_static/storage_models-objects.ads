with System;

generic
   type Typ is private;
   type Typ_Access is access all Typ;
package Storage_Models.Objects is

   type Foreign_Access is new Foreign_Address;

   function Allocate return Foreign_Access;
   function Allocate_And_Init (Src : Typ) return Foreign_Access;

   procedure Assign (Dst: Foreign_Access; Src : Typ);
   procedure Assign (Dst : in out Typ; Src : Foreign_Access);

   function Uncheck_Convert (Src : Foreign_Access) return Typ_Access;

   procedure Deallocate (Src : in out Foreign_Access);

end Storage_Models.Objects;
