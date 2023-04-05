with CUDA.Stddef;
with CUDA.Runtime_Api; use CUDA.Runtime_Api;

package body CUDA.Storage_Models is

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
      Memcpy
        (Dst   => System.Address (Target),
         Src   => Source,
         Count => CUDA.Stddef.Size_T (Size),
         Kind  => Memcpy_Host_To_Device);
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
      Memcpy
        (Dst   => Target,
         Src   => System.Address (Source),
         Count => CUDA.Stddef.Size_T (Size),
         Kind  => Memcpy_Device_To_Host);
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

   -------------------------
   -- CUDA_Async_Allocate --
   -------------------------

   procedure CUDA_Async_Allocate
     (Model           : in out CUDA_Async_Storage_Model;
      Storage_Address : out CUDA_Address;
      Size            : Storage_Count;
      Alignment       : Storage_Count)
   is
   begin
      Storage_Address := CUDA_Address (Malloc (CUDA.Stddef.Size_T (Size)));
   end CUDA_Async_Allocate;

   ---------------------------
   -- CUDA_Async_Deallocate --
   ---------------------------

   procedure CUDA_Async_Deallocate
     (Model           : in out CUDA_Async_Storage_Model;
      Storage_Address : CUDA_Address;
      Size            : Storage_Count;
      Alignment       : Storage_Count)
   is
   begin
      Free (System.Address (Storage_Address));
   end CUDA_Async_Deallocate;

   ------------------------
   -- CUDA_Async_Copy_To --
   ------------------------

   procedure CUDA_Async_Copy_To
     (Model  : in out CUDA_Async_Storage_Model;
      Target : CUDA_Address;
      Source : System.Address;
      Size   : Storage_Count)
   is
   begin
      Memcpy_Async
        (Dst   => System.Address (Target),
         Src   => Source,
         Count => CUDA.Stddef.Size_T (Size),
         Kind  => Memcpy_Host_To_Device,
         Stream => Model.Stream);
   end CUDA_Async_Copy_To;

   --------------------------
   -- CUDA_Async_Copy_From --
   --------------------------

   procedure CUDA_Async_Copy_From
     (Model  : in out CUDA_Async_Storage_Model;
      Target : System.Address;
      Source : CUDA_Address;
      Size   : Storage_Count)
   is
   begin
      Memcpy_Async
        (Dst   => Target,
         Src   => System.Address (Source),
         Count => CUDA.Stddef.Size_T (Size),
         Kind  => Memcpy_Device_To_Host,
         Stream => Model.Stream);
   end CUDA_Async_Copy_From;

   -----------------------------
   -- CUDA_Async_Storage_Size --
   -----------------------------

   function CUDA_Async_Storage_Size
     (Model : CUDA_Async_Storage_Model)
      return Storage_Count
   is
   begin
      return 0;
   end CUDA_Async_Storage_Size;

   ----------------------------
   -- CUDA_Unfified_Allocate --
   ----------------------------

   procedure CUDA_Unified_Allocate
     (Model           : in out CUDA_Async_Storage_Model;
      Storage_Address : out System.Address;
      Size            : Storage_Count;
      Alignment       : Storage_Count)
   is
   begin
      Storage_Address := Malloc_Managed (CUDA.Stddef.Size_T (Size), 1); -- cudaMemAttachGlobal
   end CUDA_Unified_Allocate;

   -----------------------------
   -- CUDA_Unified_Deallocate --
   -----------------------------

   procedure CUDA_Unified_Deallocate
     (Model           : in out CUDA_Async_Storage_Model;
      Storage_Address : System.Address;
      Size            : Storage_Count;
      Alignment       : Storage_Count)
   is
   begin
      Free (Storage_Address);
   end CUDA_Unified_Deallocate;

   ------------------------------
   -- CUDA_Pagelocked_Allocate --
   ------------------------------

   procedure CUDA_Pagelocked_Allocate
     (Model           : in out CUDA_Async_Storage_Model;
      Storage_Address : out System.Address;
      Size            : Storage_Count;
      Alignment       : Storage_Count)
   is
   begin
      Storage_Address := Malloc_Host (CUDA.Stddef.Size_T (Size));
   end CUDA_Pagelocked_Allocate;

   --------------------------------
   -- CUDA_Pagelocked_Deallocate --
   --------------------------------

   procedure CUDA_Pagelocked_Deallocate
     (Model           : in out CUDA_Async_Storage_Model;
      Storage_Address : System.Address;
      Size            : Storage_Count;
      Alignment       : Storage_Count)
   is
   begin
      Free_Host (Storage_Address);
   end CUDA_Pagelocked_Deallocate;

end CUDA.Storage_Models;
