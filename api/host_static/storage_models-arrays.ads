with System;

generic
   type Typ is private;
   type Index_Typ is (<>);
   type Array_Typ is array (Index_Typ range <>) of Typ;
   type Array_Access is access all Array_Typ;
package Storage_Models.Arrays is
   type Foreign_Array_Access is private;

   function Allocate (First, Last : Index_Typ) return Foreign_Array_Access;
   function Allocate_And_Init (Src : Array_Typ) return Foreign_Array_Access;

   procedure Assign
     (Dst : Foreign_Array_Access; Src : Array_Typ);
   procedure Assign
     (Dst : Foreign_Array_Access; First, Last : Index_Typ; Src : Array_Typ);
   procedure Assign
     (Dst : Foreign_Array_Access; Src : Typ);
   procedure Assign
     (Dst : Foreign_Array_Access; First, Last : Index_Typ; Src : Typ);
   procedure Assign
     (Dst : in out Array_Typ; Src : Foreign_Array_Access);
   procedure Assign
     (Dst : in out Array_Typ; Src : Foreign_Array_Access; First, Last : Index_Typ);

   procedure Deallocate (Src : in out Foreign_Array_Access);

   function Uncheck_Convert (Src : Foreign_Array_Access) return Array_Access;

   type Array_Typ_Bounds is record
      First, Last : Index_Typ;
   end record;

   function Bounds (Src : Foreign_Array_Access) return Array_Typ_Bounds;

private

   type Foreign_Array_Access is record
      Data   : Foreign_Address;
      Bounds : Foreign_Address;
   end record;

end Storage_Models.Arrays;
