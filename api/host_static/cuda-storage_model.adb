with CUDA.Stddef;
with CUDA.Runtime_Api; use CUDA.Runtime_Api;

package body CUDA.Storage_Model is

   -------------------
   -- CUDA_Allocate --
   -------------------

   procedure CUDA_Allocate
     (Model           : in out CUDA_Storage_Model;
      Storage_Address : out CUDA_Address;
      Size            : Storage_Count;
      Alignment       : Storage_Count)
   is
   begin
      Storage_Address := CUDA_Address (Malloc (CUDA.Stddef.Size_T (Size)));
   end CUDA_Allocate;

   ---------------------
   -- CUDA_Deallocate --
   ---------------------

   procedure CUDA_Deallocate
     (Model           : in out CUDA_Storage_Model;
      Storage_Address : CUDA_Address;
      Size            : Storage_Count;
      Alignment       : Storage_Count)
   is
   begin
      Free (System.Address (Storage_Address));
   end CUDA_Deallocate;

   ------------------
   -- CUDA_Copy_To --
   ------------------

   procedure CUDA_Copy_To
     (Model  : in out CUDA_Storage_Model;
      Target : CUDA_Address;
      Source : System.Address;
      Size   : Storage_Count)
   is
   begin
      if not Model.Async then
         Memcpy
           (Dst   => System.Address (Target),
            Src   => Source,
            Count => CUDA.Stddef.Size_T (Size),
            Kind  => Memcpy_Host_To_Device);
      else
         Memcpy_Async
           (Dst   => System.Address (Target),
            Src   => Source,
            Count => CUDA.Stddef.Size_T (Size),
            Kind  => Memcpy_Host_To_Device,
            Stream => Model.Stream);
      end if;
   end CUDA_Copy_To;

   --------------------
   -- CUDA_Copy_From --
   --------------------

   procedure CUDA_Copy_From
     (Model  : in out CUDA_Storage_Model;
      Target : System.Address;
      Source : CUDA_Address;
      Size   : Storage_Count)
   is
   begin
      if not Model.Async then
         Memcpy
           (Dst   => Target,
            Src   => System.Address (Source),
            Count => CUDA.Stddef.Size_T (Size),
            Kind  => Memcpy_Device_To_Host);
      else
         Memcpy_Async
           (Dst   => Target,
            Src   => System.Address (Source),
            Count => CUDA.Stddef.Size_T (Size),
            Kind  => Memcpy_Device_To_Host,
            Stream => Model.Stream);
      end if;
   end CUDA_Copy_From;

   -----------------------
   -- CUDA_Storage_Size --
   -----------------------

   function CUDA_Storage_Size
     (Model : CUDA_Storage_Model)
      return Storage_Count
   is
   begin
      return 0;
   end CUDA_Storage_Size;

end CUDA.Storage_Model;
