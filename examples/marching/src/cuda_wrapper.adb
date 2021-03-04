with CUDA.Runtime_Api; use CUDA.Runtime_Api;
with Interfaces.C; use Interfaces.C;
with CUDA.Driver_Types; use CUDA.Driver_Types;

package body CUDA_Wrapper is

   --------------
   -- Finalize --
   --------------

   overriding procedure Finalize (Self : in out Array_Wrapper) is 
   begin
      null;
   end Finalize;
   
   ----------
   -- From --
   ----------

   function From (From_Val : Array_T) return Array_Wrapper is   
      Ret : Array_Wrapper;
   begin
      Ret.Size := From_Val'Size / 8;
      Ret.Device_Ptr := Malloc (unsigned_long (Ret.Size)); 
      Ret.Length := From_Val'Length;
      Cuda.Runtime_Api.Memcpy
        (Dst   => Ret.Device_Ptr,
         Src   => From_Val'Address,
         Count => unsigned_long (Ret.Size),
         Kind  => Memcpy_Host_To_Device);
      return Ret;
   end From;
   
   --------
   -- To --
   --------

   procedure To (Self : Array_Wrapper; Dest : in out Array_T)is
   begin
      Cuda.Runtime_Api.Memcpy
        (Dst   => Dest'Address,
         Src   => Self.Device_Ptr,
         Count => unsigned_long (Self.Size),
         Kind  => Memcpy_Device_To_Host);
   end To;
   
   ------------
   -- Device --
   ------------

   function Device (Self : Array_Wrapper) return Array_Access is
      Ret : Array_Access (0 .. Self.Length - 1) 
        with Address => Self.Device_Ptr, Import;
   begin
      return Ret;
   end Device;
   
   ----------
   -- From --
   ----------

   function From (From_Val : T) return Wrapper is
      Ret : Wrapper;
   begin
      Ret.Size := From_Val'Size / 8;
      Ret.Device_Ptr := Malloc (unsigned_long (Ret.Size)); 
      Cuda.Runtime_Api.Memcpy
        (Dst   => Ret.Device_Ptr,
         Src   => From_Val'Address,
         Count => unsigned_long (Ret.Size), 
         Kind  => Memcpy_Host_To_Device);
      return Ret;
   end From;
   
   --------
   -- To --
   --------

   procedure To (Self : Wrapper; Dest : in out T) is
   begin
      Cuda.Runtime_Api.Memcpy
        (Dst   => Dest'Address,
         Src   => Self.Device_Ptr,
         Count => unsigned_long (Self.Size),
         Kind  => Memcpy_Device_To_Host);
   end To;
   
   ------------
   -- Device --
   ------------

   function Device (Self : Wrapper) return T_Access is
      Ret : T_Access with Address => Self.Device_Ptr, Import;
   begin
      return Ret;
   end Device;
   
   --------------
   -- Finalize --
   --------------

   overriding procedure Finalize (Self : in out Wrapper) is
   begin
      null;
   end Finalize;
   
   -----------
   -- Alloc --
   ----------- 

   procedure Reserve (Self : in out Array_Wrapper; Nb_Elements : Positive) is
   begin
      Self.Length := Nb_Elements;
      Self.Size := Self.Length * T'Size / 8;
      Self.Device_Ptr := Malloc (unsigned_long (Self.Size)); 
   end Reserve;
   
end CUDA_Wrapper;
