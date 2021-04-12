with System;

with Ada.Finalization;

generic
   type Typ is private;   
   type Index_Typ is (<>);
   type Array_Typ is array (Index_Typ range <>) of Typ;
   type Array_Access is access all Array_Typ;
package CUDA_Arrays is
   type CUDA_Array_Access is private; 

   function Allocate (First, Last : Index_Typ) return CUDA_Array_Access;   
   function Allocate_And_Init (Src : Array_Typ) return CUDA_Array_Access;

   procedure Assign 
     (Dst : CUDA_Array_Access; Src : Array_Typ);
   procedure Assign 
     (Dst : CUDA_Array_Access; First, Last : Index_Typ; Src : Array_Typ);
   procedure Assign 
     (Dst : in out Array_Typ; Src : CUDA_Array_Access);
   procedure Assign 
     (Dst : in out Array_Typ; Src : CUDA_Array_Access; First, Last : Index_Typ);
   
   procedure Deallocate (Src : in out CUDA_Array_Access);

   function Device (Src : CUDA_Array_Access) return Array_Access;      
   
   type Array_Typ_Bounds is record
      First, Last : Index_Typ;
   end record;
   
   function Bounds (Src : CUDA_Array_Access) return Array_Typ_Bounds;

private
   
   subtype CUDA_Address is System.Address;
      
   type Fat_Pointer is record
      Data   : CUDA_Address;
      Bounds : CUDA_Address;
   end record;  
   
   type CUDA_Array_Access is record
      Device_Ptr : Fat_Pointer;
   end record;   
   
end CUDA_Arrays;
