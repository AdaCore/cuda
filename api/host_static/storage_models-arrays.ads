with System;

generic
   type Typ is private;
   type Index_Typ is (<>);
   type Array_Typ is array (Index_Typ range <>) of Typ;
   type Array_Access is access all Array_Typ;
package Storage_Models.Arrays is

   type Foreign_Array_Access is record
      Data   : Foreign_Address;
      Bounds : Foreign_Address;
   end record;

   function Allocate (First, Last : Index_Typ) return Foreign_Array_Access;
   function Allocate_And_Init (Src : Array_Typ) return Foreign_Array_Access;

   procedure Assign
     (Dst : Foreign_Array_Access; Src : Array_Typ; Options : Copy_Options := Default_Copy_Options);
   procedure Assign
     (Dst : Foreign_Array_Access; First, Last : Index_Typ; Src : Array_Typ; Options : Copy_Options := Default_Copy_Options);
   procedure Assign
     (Dst : Foreign_Array_Access; Src : Typ; Options : Copy_Options := Default_Copy_Options);
   procedure Assign
     (Dst : Foreign_Array_Access; First, Last : Index_Typ; Src : Typ; Options : Copy_Options := Default_Copy_Options);
   procedure Assign
     (Dst : in out Array_Typ; Src : Foreign_Array_Access; Options : Copy_Options := Default_Copy_Options);
   procedure Assign
     (Dst : in out Array_Typ; Src : Foreign_Array_Access; First, Last : Index_Typ; Options : Copy_Options := Default_Copy_Options);

   procedure Deallocate (Src : in out Foreign_Array_Access);

   function Uncheck_Convert (Src : Foreign_Array_Access) return Array_Access;

   type Array_Typ_Bounds is record
      First, Last : Index_Typ;
   end record;

   function Bounds (Src : Foreign_Array_Access) return Array_Typ_Bounds;

   type Foreign_Array_Slice_Access is new Foreign_Array_Access;

   function Allocate (Src : Foreign_Array_Access; First, Last : Index_Typ) return Foreign_Array_Slice_Access;

   procedure Deallocate (Src : in out Foreign_Array_Slice_Access);

end Storage_Models.Arrays;
