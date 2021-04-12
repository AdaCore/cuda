with CUDA.Runtime_Api; use CUDA.Runtime_Api;
with Interfaces.C; use Interfaces.C;
with CUDA.Driver_Types; use CUDA.Driver_Types;

package body CUDA_Objects is
   
   --------------
   -- Allocate --
   --------------

   function Allocate return CUDA_Access is
      Ret : CUDA_Access;
   begin
      Ret.Device_Ptr := Malloc (Typ'Size / 8);
      
      return Ret;
   end Allocate;
   
   -----------------------
   -- Allocate_And_Init --
   -----------------------
   
   function Allocate_And_Init (Src : Typ) return CUDA_Access is
      Ret : CUDA_Access := Allocate;
   begin
      Assign (Ret, Src);
      
      return Ret;
   end Allocate_And_Init;
   
   ------------
   -- Assign --
   ------------
   
   procedure Assign (Dst: CUDA_Access; Src : Typ) is
   begin
      Cuda.Runtime_Api.Memcpy
        (Dst   => Dst.Device_Ptr,
         Src   => Src'Address,
         Count => Typ'Size / 8,
         Kind  => Memcpy_Host_To_Device);
   end Assign;
   
   ------------
   -- Assign --
   ------------
   
   procedure Assign (Dst : in out Typ; Src : CUDA_Access) is
   begin
      Cuda.Runtime_Api.Memcpy
        (Dst   => Dst'Address,
         Src   => Src.Device_Ptr,
         Count => Typ'Size / 8,
         Kind  => Memcpy_Device_To_Host);      
   end Assign;
   
   ------------
   -- Device --
   ------------
     
   function Device (Src : CUDA_Access) return Typ_Access is
      Ret : Typ_Access with Address => Src.Device_Ptr'Address, Import;
   begin
      return Ret;
   end Device;   
   
   ----------------
   -- Deallocate --
   ----------------
   
   procedure Deallocate (Src : in out CUDA_Access) is
   begin
      Free (Src.Device_Ptr);
   end Deallocate;
   
end CUDA_Objects;
