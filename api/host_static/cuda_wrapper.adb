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
      --  Free (Self.Device_Ptr.Data);
      --  Free (Self.Device_Ptr.Bounds);
   end Finalize;
   
   ----------
   -- From --
   ----------

   function From (From_Val : Array_T) return Array_Wrapper is         
      Ret : Array_Wrapper;
      From_Bounds : aliased T_Bounds := (From_Val'First, From_Val'Last);
   begin      
      Ret.Device_Ptr.Data := Malloc (From_Val'Size / 8);
      Ret.Device_Ptr.Bounds := Malloc (T_Bounds'Size / 8);

      Cuda.Runtime_Api.Memcpy
        (Dst   => Ret.Device_Ptr.Data,
         Src   => From_Val (From_Val'First)'Address,
         Count => From_Val'Size / 8,
         Kind  => Memcpy_Host_To_Device);
      
      Cuda.Runtime_Api.Memcpy
        (Dst   => Ret.Device_Ptr.Bounds,
         Src   => From_Bounds'Address,
         Count => T_Bounds'Size / 8,
         Kind  => Memcpy_Host_To_Device);
      
      return Ret;
   end From;
   
   --------
   -- To --
   --------

   procedure To (Self : Array_Wrapper; Dest : in out Array_T)is
      From_Bounds : aliased T_Bounds;
   begin
      Cuda.Runtime_Api.Memcpy
        (Dst   => From_Bounds'Address,
         Src   => Self.Device_Ptr.Bounds,
         Count => T_Bounds'Size / 8,
         Kind  => Memcpy_Device_To_Host);
      
      if Dest'Length /= From_Bounds.Last - From_Bounds.First + 1 then
         raise Constraint_Error;
      end if;
      
      Cuda.Runtime_Api.Memcpy
        (Dst   => Dest'Address,
         Src   => Self.Device_Ptr.Data,
         Count => Dest'Size / 8,
         Kind  => Memcpy_Device_To_Host);
   end To;
   
   ------------
   -- Device --
   ------------

   function Device (Self : Array_Wrapper) return Array_Access is
      Ret : Array_Access with Address => Self.Device_Ptr'Address, Import;
   begin
      return Ret;
   end Device;
   
   ----------
   -- From --
   ----------

   function From (From_Val : T) return Wrapper is
      Ret : Wrapper;
   begin
      Ret.Device_Ptr := Malloc (T'Size / 8); 
      
      Cuda.Runtime_Api.Memcpy
        (Dst   => Ret.Device_Ptr,
         Src   => From_Val'Address,
         Count => T'Size / 8, 
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
         Count => T'Size / 8,
         Kind  => Memcpy_Device_To_Host);
   end To;
   
   ---------
   -- Get --
   ---------
   
   function Get (Self : Wrapper) return T is
      Result : aliased T;
   begin
      Cuda.Runtime_Api.Memcpy
        (Dst   => Result'Address,
         Src   => Self.Device_Ptr,
         Count => T'Size / 8,
         Kind  => Memcpy_Device_To_Host);
      
      return Result;
   end Get;
   
   ------------
   -- Device --
   ------------

   function Device (Self : Wrapper) return T_Access is
      Ret : T_Access with Address => Self.Device_Ptr'Address, Import;
   begin
      return Ret;
   end Device;
   
   --------------
   -- Finalize --
   --------------

   overriding procedure Finalize (Self : in out Wrapper) is
   begin
      null;
      --Free (Self.Device_Ptr);
   end Finalize;
   
   -----------
   -- Alloc --
   ----------- 

   procedure Reserve (Self : in out Array_Wrapper; First, Last : Natural) is     
      Bounds : aliased T_Bounds := (First, Last);
   begin      
      Self.Device_Ptr.Data := Malloc (unsigned_long (Last - First + 1) * T'Size / 8); 
      Self.Device_Ptr.Bounds := Malloc (T_Bounds'Size / 8);      
      
      Cuda.Runtime_Api.Memcpy
        (Dst   => Self.Device_Ptr.Bounds,
         Src   => Bounds'Address,
         Count => T_Bounds'Size / 8,
         Kind  => Memcpy_Host_To_Device);
   end Reserve;
   
end CUDA_Wrapper;
