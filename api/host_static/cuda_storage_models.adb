with System.Storage_Elements; use System.Storage_Elements;

with CUDA.Stddef;
with CUDA.Runtime_Api; use CUDA.Runtime_Api;
with CUDA.Driver_Types; use CUDA.Driver_Types;

package body CUDA_Storage_Models is

   function Address_Offset (Address : System.Address; Bytes : Natural) return System.Address is
   begin
      return Address + Storage_Offset (Bytes);
   end Address_Offset;

   function Malloc_Allocate (Size : Natural) return System.Address is
   begin
      return Malloc (CUDA.Stddef.Size_T (Size));
   end Malloc_Allocate;

   procedure Malloc_Deallocate (Address : System.Address) is
   begin
      Free (Address);
   end Malloc_Deallocate;

   procedure Malloc_Copy_To_Foreign (Dst : System.Address; Src : System.Address; Bytes : Natural; Options : Copy_Options) is
   begin
      if not Options.Async then
         Memcpy
           (Dst   => Dst,
            Src   => Src,
            Count => CUDA.Stddef.Size_T (Bytes),
            Kind  => Memcpy_Host_To_Device);
      else
         Memcpy_Async
           (Dst   => Dst,
            Src   => Src,
            Count => CUDA.Stddef.Size_T (Bytes),
            Kind  => Memcpy_Host_To_Device,
            Stream => Options.Stream);
      end if;
   end Malloc_Copy_To_Foreign;

   procedure Malloc_Copy_To_Native (Dst : System.Address; Src : System.Address; Bytes : Natural; Options : Copy_Options) is
   begin
      if not Options.Async then
         Memcpy
           (Dst   => Dst,
            Src   => Src,
            Count => CUDA.Stddef.Size_T (Bytes),
            Kind  => Memcpy_Device_To_Host);
      else
         Memcpy_Async
           (Dst   => Dst,
            Src   => Src,
            Count => CUDA.Stddef.Size_T (Bytes),
            Kind  => Memcpy_Device_To_Host,
            Stream => Options.Stream);
      end if;
   end Malloc_Copy_To_Native;

   function Malloc_Host_Allocate (Size : Natural) return System.Address is
   begin
      return Malloc_Host (CUDA.Stddef.Size_T (Size));
   end Malloc_Host_Allocate;

   procedure Malloc_Host_Deallocate (Address : System.Address) is
   begin
      Free_Host (Address);
   end Malloc_Host_Deallocate;

   procedure Malloc_Host_Copy_To_Foreign (Dst : System.Address; Src : System.Address; Bytes : Natural; Options : Copy_Options) is
   begin
      if not Options.Async then
         Memcpy
           (Dst   => Dst,
            Src   => Src,
            Count => CUDA.Stddef.Size_T (Bytes),
            Kind  => Memcpy_Host_To_Device);
      else
         Memcpy_Async
           (Dst   => Dst,
            Src   => Src,
            Count => CUDA.Stddef.Size_T (Bytes),
            Kind  => Memcpy_Host_To_Device,
            Stream => Options.Stream);
      end if;
   end Malloc_Host_Copy_To_Foreign;

   procedure Malloc_Host_Copy_To_Native (Dst : System.Address; Src : System.Address; Bytes : Natural; Options : Copy_Options) is
   begin
      if not Options.Async then
         Memcpy
           (Dst   => Dst,
            Src   => Src,
            Count => CUDA.Stddef.Size_T (Bytes),
            Kind  => Memcpy_Device_To_Host);
      else
         Memcpy_Async
           (Dst   => Dst,
            Src   => Src,
            Count => CUDA.Stddef.Size_T (Bytes),
            Kind  => Memcpy_Device_To_Host,
            Stream => Options.Stream);
      end if;
   end Malloc_Host_Copy_To_Native;

end CUDA_Storage_Models;
