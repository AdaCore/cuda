with CUDA.Runtime_Api; use CUDA.Runtime_Api;
with Interfaces.C; use Interfaces.C;
with CUDA.Driver_Types; use CUDA.Driver_Types;

package body CUDA_Arrays is
   
   function Length (First, Last : Index_Typ) return unsigned_long is
      (Index_Typ'Pos (Last) - Index_Typ'Pos (First) + 1);

   --------------
   -- Allocate --
   --------------

   function Allocate (First, Last : Index_Typ) return CUDA_Array_Access is     
      Ret : CUDA_Array_Access;
      Bounds : aliased Array_Typ_Bounds := (First, Last);
   begin      
      Ret.Device_Ptr.Data := Malloc (Length (First, Last) * Typ'Size / 8); 
      Ret.Device_Ptr.Bounds := Malloc (Array_Typ_Bounds'Size / 8);      
      
      Cuda.Runtime_Api.Memcpy
        (Dst   => Ret.Device_Ptr.Bounds,
         Src   => Bounds'Address,
         Count => Array_Typ_Bounds'Size / 8,
         Kind  => Memcpy_Host_To_Device);
      
      return Ret;
   end Allocate;
   
   -----------------------
   -- Allocate_And_Init --
   -----------------------

   function Allocate_And_Init (Src : Array_Typ) return CUDA_Array_Access is         
      Ret : CUDA_Array_Access := Allocate (Src'First, Src'Last);      
   begin      
      Assign (Ret, Src);
      
      return Ret;
   end Allocate_And_Init;
   
   ----------
   -- Assign --
   ---------
   
   procedure Assign (Dst : CUDA_Array_Access; Src : Array_Typ) is
      Dst_Bounds : Array_Typ_Bounds := Bounds (Dst);
   begin
      Assign (Dst, Dst_Bounds.First, Dst_Bounds.Last, Src);
   end Assign;
   
   ----------
   -- Assign --
   ----------
   
   procedure Assign (Dst : CUDA_Array_Access; First, Last : Index_Typ; Src : Array_Typ) is      
      Dst_Length : unsigned_long := Length (First, Last);
   begin      
      if Dst_Length /= Src'Length then
         raise Constraint_Error;
      end if;      
      
      Cuda.Runtime_Api.Memcpy
        (Dst   => Dst.Device_Ptr.Data,
         Src   => Src'Address,
         Count => Dst_Length * Typ'Size / 8,
         Kind  => Memcpy_Host_To_Device);
   end Assign;
   
   ------------
   -- Assign --
   ------------

   procedure Assign (Dst : in out Array_Typ; Src : CUDA_Array_Access) is
      Src_Bounds : Array_Typ_Bounds := Bounds (Src);
   begin
      Assign (Dst, Src, Src_Bounds.First, Src_Bounds.Last);
   end Assign;

   ------------
   -- Assign --
   ------------
   
   procedure Assign (Dst : in out Array_Typ; Src : CUDA_Array_Access; First, Last : Index_Typ) is
       Src_Length : unsigned_long := Length (First, Last);
   begin
      if Src_Length /= Dst'Length then
         raise Constraint_Error;
      end if;
      
      Cuda.Runtime_Api.Memcpy
        (Dst   => Dst'Address,
         Src   => Src.Device_Ptr.Data,
         Count => Src_Length * Typ'Size / 8,
         Kind  => Memcpy_Device_To_Host);
   end Assign;
   
   ------------
   -- Device --
   ------------

   function Device (Src : CUDA_Array_Access) return Array_Access is
      Ret : Array_Access with Address => Src.Device_Ptr'Address, Import;
   begin
      return Ret;
   end Device;   
   
   ------------
   -- Bounds --
   ------------
   
   function Bounds (Src : CUDA_Array_Access) return Array_Typ_Bounds is
       Bounds : aliased Array_Typ_Bounds;
   begin
      Cuda.Runtime_Api.Memcpy
        (Dst   => Bounds'Address,
         Src   => Src.Device_Ptr.Bounds,
         Count => Array_Typ_Bounds'Size / 8,
         Kind  => Memcpy_Device_To_Host);
      
      return Bounds;
   end Bounds;
      
   procedure Deallocate (Src : in out CUDA_Array_Access) is
   begin
      Free (Src.Device_Ptr.Data);
      Free (Src.Device_Ptr.Bounds);
   end Deallocate;
      
end CUDA_Arrays;
