with System;

generic
   type Typ is private;   
   type Typ_Access is access all Typ;
package CUDA_Objects is
   
   type CUDA_Access is private;
   
   function Allocate return CUDA_Access;
   function Allocate_And_Init (Src : Typ) return CUDA_Access;
   
   procedure Assign (Dst: CUDA_Access; Src : Typ);
   procedure Assign (Dst : in out Typ; Src : CUDA_Access);
     
   function Device (Src : CUDA_Access) return Typ_Access;

   procedure Deallocate (Src : in out CUDA_Access);
private
   
   subtype CUDA_Address is System.Address;
   
   type CUDA_Access is record
      Device_Ptr : CUDA_Address;     
   end record;
   
end CUDA_Objects;
