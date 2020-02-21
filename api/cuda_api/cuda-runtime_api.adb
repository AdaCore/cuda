with Ada.Exceptions;
with Interfaces.C.Strings;
with crtdefs_h;
with cuda_runtime_api_h;
with driver_types_h;
with surface_types_h;
with texture_types_h;
with vector_types_h;
use cuda_runtime_api_h;

package body CUDA.Runtime_Api is

   overriding procedure Allocate (Self : in out CUDA_Device_Pool; Addr : out System.Address; Size : System.Storage_Elements.Storage_Count; Alignment : System.Storage_Elements.Storage_Count) is
   begin
      Addr := Malloc (CUDA.Crtdefs.Size_T (Size));
   end Allocate;

   overriding procedure Copy_To_Pool (Self : in out CUDA_Device_Pool; Addr : System.Address; Value : aliased System.Storage_Elements.Storage_Array; Size : System.Storage_Elements.Storage_Count) is
   begin
      Memcpy (Addr, Value'Address, CUDA.Crtdefs.Size_T (Size), CUDA.Driver_Types.Memcpy_Host_To_Device);
   end Copy_To_Pool;

   overriding procedure Deallocate (Self : in out CUDA_Device_Pool; Addr : System.Address; Size : System.Storage_Elements.Storage_Count; Alignment : System.Storage_Elements.Storage_Count) is
   begin
      Free (Addr);
   end Deallocate;

   function Grid_Dim return CUDA.Vector_Types.Dim3 is

      function Nctaid_X return Interfaces.C.unsigned with
         Inline,
         Import,
         Convention    => Builtin,
         External_Name => "nvvm.read.ptx.sreg.ntaid.x";
      function Nctaid_Y return Interfaces.C.unsigned with
         Inline,
         Import,
         Convention    => Builtin,
         External_Name => "nvvm.read.ptx.sreg.ntaid.y";
      function Nctaid_Z return Interfaces.C.unsigned with
         Inline,
         Import,
         Convention    => Builtin,
         External_Name => "nvvm.read.ptx.sreg.ntaid.z";

   begin
      return (Nctaid_X, Nctaid_Y, Nctaid_Z);
   end Grid_Dim;

   function Block_Idx return CUDA.Vector_Types.Uint3 is

      function Ctaid_X return Interfaces.C.unsigned with
         Inline,
         Import,
         Convention    => Builtin,
         External_Name => "nvvm.read.ptx.sreg.ctid.x";
      function Ctaid_Y return Interfaces.C.unsigned with
         Inline,
         Import,
         Convention    => Builtin,
         External_Name => "nvvm.read.ptx.sreg.ctid.y";
      function Ctaid_Z return Interfaces.C.unsigned with
         Inline,
         Import,
         Convention    => Builtin,
         External_Name => "nvvm.read.ptx.sreg.ctid.z";

   begin
      return (Ctaid_X, Ctaid_Y, Ctaid_Z);
   end Block_Idx;

   function Block_Dim return CUDA.Vector_Types.Dim3 is

      function Ntid_X return Interfaces.C.unsigned with
         Inline,
         Import,
         Convention    => Builtin,
         External_Name => "nvvm.read.ptx.sreg.ntid.x";
      function Ntid_Y return Interfaces.C.unsigned with
         Inline,
         Import,
         Convention    => Builtin,
         External_Name => "nvvm.read.ptx.sreg.ntid.y";
      function Ntid_Z return Interfaces.C.unsigned with
         Inline,
         Import,
         Convention    => Builtin,
         External_Name => "nvvm.read.ptx.sreg.ntid.z";

   begin
      return (Ntid_X, Ntid_Y, Ntid_Z);
   end Block_Dim;

   function Thread_Idx return CUDA.Vector_Types.Uint3 is

      function Tid_X return Interfaces.C.unsigned with
         Inline,
         Import,
         Convention    => Builtin,
         External_Name => "nvvm.read.ptx.sreg.tid.x";
      function Tid_Y return Interfaces.C.unsigned with
         Inline,
         Import,
         Convention    => Builtin,
         External_Name => "nvvm.read.ptx.sreg.tid.y";
      function Tid_Z return Interfaces.C.unsigned with
         Inline,
         Import,
         Convention    => Builtin,
         External_Name => "nvvm.read.ptx.sreg.tid.z";

   begin
      return (Tid_X, Tid_Y, Tid_Z);
   end Thread_Idx;

   function Wrap_Size return Interfaces.C.int is

      function Wrapsize return Interfaces.C.int with
         Inline,
         Import,
         Convention    => Builtin,
         External_Name => "nvvm.read.ptx.sreg.wrapsize";

   begin
      return Wrapsize;
   end Wrap_Size;

   procedure Device_Reset is

      Local_Tmp_0 : driver_types_h.cudaError_t  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/cuda_runtime_api.h:280
      := cuda_runtime_api_h.cudaDeviceReset;

   begin
      if Local_Tmp_0 /= 0 then
         Ada.Exceptions.Raise_Exception (CUDA.Exception_Registry.Element (Integer (Local_Tmp_0)));
      end if;
   end Device_Reset;

   procedure Device_Synchronize is

      Local_Tmp_0 : driver_types_h.cudaError_t  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/cuda_runtime_api.h:301
      := cuda_runtime_api_h.cudaDeviceSynchronize;

   begin
      if Local_Tmp_0 /= 0 then
         Ada.Exceptions.Raise_Exception (CUDA.Exception_Registry.Element (Integer (Local_Tmp_0)));
      end if;
   end Device_Synchronize;

   procedure Device_Set_Limit (Limit : CUDA.Driver_Types.Limit; Value : CUDA.Crtdefs.Size_T) is

      Local_Tmp_1 : driver_types_h.cudaLimit with
         Address => Limit'Address,
         Import;
      Local_Tmp_2 : crtdefs_h.size_t with
         Address => Value'Address,
         Import;
      Local_Tmp_0 : driver_types_h.cudaError_t  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/cuda_runtime_api.h:386
      := cuda_runtime_api_h.cudaDeviceSetLimit (Local_Tmp_1, Local_Tmp_2);

   begin
      if Local_Tmp_0 /= 0 then
         Ada.Exceptions.Raise_Exception (CUDA.Exception_Registry.Element (Integer (Local_Tmp_0)));
      end if;
   end Device_Set_Limit;

   function Device_Get_Limit (Limit : CUDA.Driver_Types.Limit) return CUDA.Crtdefs.Size_T is

      Local_Tmp_1 : aliased crtdefs_h.size_t;
      Local_Tmp_2 : aliased CUDA.Crtdefs.Size_T with
         Address => Local_Tmp_1'Address,
         Import;
      Local_Tmp_3 : driver_types_h.cudaLimit with
         Address => Limit'Address,
         Import;
      Local_Tmp_0 : driver_types_h.cudaError_t  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/cuda_runtime_api.h:420
      := cuda_runtime_api_h.cudaDeviceGetLimit (Local_Tmp_1'Unchecked_Access, Local_Tmp_3);

   begin
      if Local_Tmp_0 /= 0 then
         Ada.Exceptions.Raise_Exception (CUDA.Exception_Registry.Element (Integer (Local_Tmp_0)));
      end if;
      return Local_Tmp_2;
   end Device_Get_Limit;

   function Device_Get_Cache_Config return CUDA.Driver_Types.Func_Cache is

      Local_Tmp_1 : aliased driver_types_h.cudaFuncCache;
      Local_Tmp_2 : aliased CUDA.Driver_Types.Func_Cache with
         Address => Local_Tmp_1'Address,
         Import;
      Local_Tmp_0 : driver_types_h.cudaError_t  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/cuda_runtime_api.h:453
      := cuda_runtime_api_h.cudaDeviceGetCacheConfig (Local_Tmp_1'Unchecked_Access);

   begin
      if Local_Tmp_0 /= 0 then
         Ada.Exceptions.Raise_Exception (CUDA.Exception_Registry.Element (Integer (Local_Tmp_0)));
      end if;
      return Local_Tmp_2;
   end Device_Get_Cache_Config;

   function Device_Get_Stream_Priority_Range (Greatest_Priority : out int) return int is

      Local_Tmp_1 : aliased int;
      Local_Tmp_2 : aliased int with
         Address => Local_Tmp_1'Address,
         Import;
      Local_Tmp_3 : aliased int with
         Address => Greatest_Priority'Address,
         Import;
      Local_Tmp_0 : driver_types_h.cudaError_t  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/cuda_runtime_api.h:490
      := cuda_runtime_api_h.cudaDeviceGetStreamPriorityRange (Local_Tmp_1'Unchecked_Access, Local_Tmp_3'Access);

   begin
      if Local_Tmp_0 /= 0 then
         Ada.Exceptions.Raise_Exception (CUDA.Exception_Registry.Element (Integer (Local_Tmp_0)));
      end if;
      return Local_Tmp_2;
   end Device_Get_Stream_Priority_Range;

   procedure Device_Set_Cache_Config (Cache_Config : CUDA.Driver_Types.Func_Cache) is

      Local_Tmp_1 : driver_types_h.cudaFuncCache with
         Address => Cache_Config'Address,
         Import;
      Local_Tmp_0 : driver_types_h.cudaError_t  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/cuda_runtime_api.h:534
      := cuda_runtime_api_h.cudaDeviceSetCacheConfig (Local_Tmp_1);

   begin
      if Local_Tmp_0 /= 0 then
         Ada.Exceptions.Raise_Exception (CUDA.Exception_Registry.Element (Integer (Local_Tmp_0)));
      end if;
   end Device_Set_Cache_Config;

   function Device_Get_Shared_Mem_Config return CUDA.Driver_Types.Shared_Mem_Config is

      Local_Tmp_1 : aliased driver_types_h.cudaSharedMemConfig;
      Local_Tmp_2 : aliased CUDA.Driver_Types.Shared_Mem_Config with
         Address => Local_Tmp_1'Address,
         Import;
      Local_Tmp_0 : driver_types_h.cudaError_t  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/cuda_runtime_api.h:565
      := cuda_runtime_api_h.cudaDeviceGetSharedMemConfig (Local_Tmp_1'Unchecked_Access);

   begin
      if Local_Tmp_0 /= 0 then
         Ada.Exceptions.Raise_Exception (CUDA.Exception_Registry.Element (Integer (Local_Tmp_0)));
      end if;
      return Local_Tmp_2;
   end Device_Get_Shared_Mem_Config;

   procedure Device_Set_Shared_Mem_Config (Config : CUDA.Driver_Types.Shared_Mem_Config) is

      Local_Tmp_1 : driver_types_h.cudaSharedMemConfig with
         Address => Config'Address,
         Import;
      Local_Tmp_0 : driver_types_h.cudaError_t  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/cuda_runtime_api.h:609
      := cuda_runtime_api_h.cudaDeviceSetSharedMemConfig (Local_Tmp_1);

   begin
      if Local_Tmp_0 /= 0 then
         Ada.Exceptions.Raise_Exception (CUDA.Exception_Registry.Element (Integer (Local_Tmp_0)));
      end if;
   end Device_Set_Shared_Mem_Config;

   function Device_Get_By_PCIBus_Id (Pci_Bus_Id : String) return int is

      Local_Tmp_1 : aliased int;
      Local_Tmp_2 : aliased int with
         Address => Local_Tmp_1'Address,
         Import;
      Local_Tmp_3 : Interfaces.C.Strings.chars_ptr := Interfaces.C.Strings.New_String (Pci_Bus_Id);
      Local_Tmp_0 : driver_types_h.cudaError_t  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/cuda_runtime_api.h:636
      := cuda_runtime_api_h.cudaDeviceGetByPCIBusId (Local_Tmp_1'Unchecked_Access, Local_Tmp_3);

   begin
      if Local_Tmp_0 /= 0 then
         Ada.Exceptions.Raise_Exception (CUDA.Exception_Registry.Element (Integer (Local_Tmp_0)));
      end if;
      Interfaces.C.Strings.Free (Local_Tmp_3);
      return Local_Tmp_2;
   end Device_Get_By_PCIBus_Id;

   procedure Device_Get_PCIBus_Id (Pci_Bus_Id : String; Len : int; Device : int) is

      Local_Tmp_1 : Interfaces.C.Strings.chars_ptr := Interfaces.C.Strings.New_String (Pci_Bus_Id);
      Local_Tmp_2 : int with
         Address => Len'Address,
         Import;
      Local_Tmp_3 : int with
         Address => Device'Address,
         Import;
      Local_Tmp_0 : driver_types_h.cudaError_t  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/cuda_runtime_api.h:666
      := cuda_runtime_api_h.cudaDeviceGetPCIBusId (Local_Tmp_1, Local_Tmp_2, Local_Tmp_3);

   begin
      if Local_Tmp_0 /= 0 then
         Ada.Exceptions.Raise_Exception (CUDA.Exception_Registry.Element (Integer (Local_Tmp_0)));
      end if;
      Interfaces.C.Strings.Free (Local_Tmp_1);
   end Device_Get_PCIBus_Id;

   function Ipc_Get_Event_Handle (Event : CUDA.Driver_Types.Event_T) return CUDA.Driver_Types.Ipc_Event_Handle_T is

      Local_Tmp_1 : aliased driver_types_h.cudaIpcEventHandle_t;
      Local_Tmp_2 : aliased CUDA.Driver_Types.Ipc_Event_Handle_T with
         Address => Local_Tmp_1'Address,
         Import;
      Local_Tmp_3 : driver_types_h.cudaEvent_t with
         Address => Event'Address,
         Import;
      Local_Tmp_0 : driver_types_h.cudaError_t  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/cuda_runtime_api.h:713
      := cuda_runtime_api_h.cudaIpcGetEventHandle (Local_Tmp_1'Unchecked_Access, Local_Tmp_3);

   begin
      if Local_Tmp_0 /= 0 then
         Ada.Exceptions.Raise_Exception (CUDA.Exception_Registry.Element (Integer (Local_Tmp_0)));
      end if;
      return Local_Tmp_2;
   end Ipc_Get_Event_Handle;

   function Ipc_Open_Event_Handle (Handle : CUDA.Driver_Types.Ipc_Event_Handle_T) return CUDA.Driver_Types.Event_T is

      Local_Tmp_1 : aliased CUDA.Driver_Types.Event_T;
      Local_Tmp_2 : driver_types_h.cudaIpcEventHandle_t with
         Address => Handle'Address,
         Import;
      Local_Tmp_0 : driver_types_h.cudaError_t  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/cuda_runtime_api.h:753
      := cuda_runtime_api_h.cudaIpcOpenEventHandle (Local_Tmp_1'Address, Local_Tmp_2);

   begin
      if Local_Tmp_0 /= 0 then
         Ada.Exceptions.Raise_Exception (CUDA.Exception_Registry.Element (Integer (Local_Tmp_0)));
      end if;
      return Local_Tmp_1;
   end Ipc_Open_Event_Handle;

   function Ipc_Get_Mem_Handle (Dev_Ptr : System.Address) return CUDA.Driver_Types.Ipc_Mem_Handle_T is

      Local_Tmp_1 : aliased driver_types_h.cudaIpcMemHandle_t;
      Local_Tmp_2 : aliased CUDA.Driver_Types.Ipc_Mem_Handle_T with
         Address => Local_Tmp_1'Address,
         Import;
      Local_Tmp_3 : System.Address with
         Address => Dev_Ptr'Address,
         Import;
      Local_Tmp_0 : driver_types_h.cudaError_t  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/cuda_runtime_api.h:796
      := cuda_runtime_api_h.cudaIpcGetMemHandle (Local_Tmp_1'Unchecked_Access, Local_Tmp_3);

   begin
      if Local_Tmp_0 /= 0 then
         Ada.Exceptions.Raise_Exception (CUDA.Exception_Registry.Element (Integer (Local_Tmp_0)));
      end if;
      return Local_Tmp_2;
   end Ipc_Get_Mem_Handle;

   procedure Ipc_Open_Mem_Handle (Dev_Ptr : System.Address; Handle : CUDA.Driver_Types.Ipc_Mem_Handle_T; Flags : unsigned) is

      Local_Tmp_1 : System.Address with
         Address => Dev_Ptr'Address,
         Import;
      Local_Tmp_2 : driver_types_h.cudaIpcMemHandle_t with
         Address => Handle'Address,
         Import;
      Local_Tmp_3 : unsigned with
         Address => Flags'Address,
         Import;
      Local_Tmp_0 : driver_types_h.cudaError_t  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/cuda_runtime_api.h:854
      := cuda_runtime_api_h.cudaIpcOpenMemHandle (Local_Tmp_1, Local_Tmp_2, Local_Tmp_3);

   begin
      if Local_Tmp_0 /= 0 then
         Ada.Exceptions.Raise_Exception (CUDA.Exception_Registry.Element (Integer (Local_Tmp_0)));
      end if;
   end Ipc_Open_Mem_Handle;

   procedure Ipc_Close_Mem_Handle (Dev_Ptr : System.Address) is

      Local_Tmp_1 : System.Address with
         Address => Dev_Ptr'Address,
         Import;
      Local_Tmp_0 : driver_types_h.cudaError_t  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/cuda_runtime_api.h:889
      := cuda_runtime_api_h.cudaIpcCloseMemHandle (Local_Tmp_1);

   begin
      if Local_Tmp_0 /= 0 then
         Ada.Exceptions.Raise_Exception (CUDA.Exception_Registry.Element (Integer (Local_Tmp_0)));
      end if;
   end Ipc_Close_Mem_Handle;

   procedure Thread_Exit is

      Local_Tmp_0 : driver_types_h.cudaError_t  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/cuda_runtime_api.h:931
      := cuda_runtime_api_h.cudaThreadExit;

   begin
      if Local_Tmp_0 /= 0 then
         Ada.Exceptions.Raise_Exception (CUDA.Exception_Registry.Element (Integer (Local_Tmp_0)));
      end if;
   end Thread_Exit;

   procedure Thread_Synchronize is

      Local_Tmp_0 : driver_types_h.cudaError_t  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/cuda_runtime_api.h:957
      := cuda_runtime_api_h.cudaThreadSynchronize;

   begin
      if Local_Tmp_0 /= 0 then
         Ada.Exceptions.Raise_Exception (CUDA.Exception_Registry.Element (Integer (Local_Tmp_0)));
      end if;
   end Thread_Synchronize;

   procedure Thread_Set_Limit (Limit : CUDA.Driver_Types.Limit; Value : CUDA.Crtdefs.Size_T) is

      Local_Tmp_1 : driver_types_h.cudaLimit with
         Address => Limit'Address,
         Import;
      Local_Tmp_2 : crtdefs_h.size_t with
         Address => Value'Address,
         Import;
      Local_Tmp_0 : driver_types_h.cudaError_t  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/cuda_runtime_api.h:1006
      := cuda_runtime_api_h.cudaThreadSetLimit (Local_Tmp_1, Local_Tmp_2);

   begin
      if Local_Tmp_0 /= 0 then
         Ada.Exceptions.Raise_Exception (CUDA.Exception_Registry.Element (Integer (Local_Tmp_0)));
      end if;
   end Thread_Set_Limit;

   function Thread_Get_Limit (Limit : CUDA.Driver_Types.Limit) return CUDA.Crtdefs.Size_T is

      Local_Tmp_1 : aliased crtdefs_h.size_t;
      Local_Tmp_2 : aliased CUDA.Crtdefs.Size_T with
         Address => Local_Tmp_1'Address,
         Import;
      Local_Tmp_3 : driver_types_h.cudaLimit with
         Address => Limit'Address,
         Import;
      Local_Tmp_0 : driver_types_h.cudaError_t  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/cuda_runtime_api.h:1039
      := cuda_runtime_api_h.cudaThreadGetLimit (Local_Tmp_1'Unchecked_Access, Local_Tmp_3);

   begin
      if Local_Tmp_0 /= 0 then
         Ada.Exceptions.Raise_Exception (CUDA.Exception_Registry.Element (Integer (Local_Tmp_0)));
      end if;
      return Local_Tmp_2;
   end Thread_Get_Limit;

   function Thread_Get_Cache_Config return CUDA.Driver_Types.Func_Cache is

      Local_Tmp_1 : aliased driver_types_h.cudaFuncCache;
      Local_Tmp_2 : aliased CUDA.Driver_Types.Func_Cache with
         Address => Local_Tmp_1'Address,
         Import;
      Local_Tmp_0 : driver_types_h.cudaError_t  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/cuda_runtime_api.h:1075
      := cuda_runtime_api_h.cudaThreadGetCacheConfig (Local_Tmp_1'Unchecked_Access);

   begin
      if Local_Tmp_0 /= 0 then
         Ada.Exceptions.Raise_Exception (CUDA.Exception_Registry.Element (Integer (Local_Tmp_0)));
      end if;
      return Local_Tmp_2;
   end Thread_Get_Cache_Config;

   procedure Thread_Set_Cache_Config (Cache_Config : CUDA.Driver_Types.Func_Cache) is

      Local_Tmp_1 : driver_types_h.cudaFuncCache with
         Address => Cache_Config'Address,
         Import;
      Local_Tmp_0 : driver_types_h.cudaError_t  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/cuda_runtime_api.h:1122
      := cuda_runtime_api_h.cudaThreadSetCacheConfig (Local_Tmp_1);

   begin
      if Local_Tmp_0 /= 0 then
         Ada.Exceptions.Raise_Exception (CUDA.Exception_Registry.Element (Integer (Local_Tmp_0)));
      end if;
   end Thread_Set_Cache_Config;

   function Get_Last_Error return CUDA.Driver_Types.Error_T is

      Local_Tmp_1 : driver_types_h.cudaError_t  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/cuda_runtime_api.h:1181
      := cuda_runtime_api_h.cudaGetLastError;
      Local_Tmp_0 : CUDA.Driver_Types.Error_T with
         Address => Local_Tmp_1'Address,
         Import;

   begin
      return Local_Tmp_0;
   end Get_Last_Error;

   function Peek_At_Last_Error return CUDA.Driver_Types.Error_T is

      Local_Tmp_1 : driver_types_h.cudaError_t  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/cuda_runtime_api.h:1227
      := cuda_runtime_api_h.cudaPeekAtLastError;
      Local_Tmp_0 : CUDA.Driver_Types.Error_T with
         Address => Local_Tmp_1'Address,
         Import;

   begin
      return Local_Tmp_0;
   end Peek_At_Last_Error;

   function Get_Error_Name (Arg1 : CUDA.Driver_Types.Error_T) return Interfaces.C.Strings.chars_ptr  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/cuda_runtime_api.h:1243
   is

      Local_Tmp_2 : driver_types_h.cudaError_t with
         Address => Arg1'Address,
         Import;
      Local_Tmp_1 : Interfaces.C.Strings.chars_ptr  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/cuda_runtime_api.h:1243
      := cuda_runtime_api_h.cudaGetErrorName (Local_Tmp_2);
      Local_Tmp_0 : Interfaces.C.Strings.chars_ptr with
         Address => Local_Tmp_1'Address,
         Import;

   begin
      return Local_Tmp_0;
   end Get_Error_Name;

   function Get_Error_String (Arg1 : CUDA.Driver_Types.Error_T) return Interfaces.C.Strings.chars_ptr  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/cuda_runtime_api.h:1259
   is

      Local_Tmp_2 : driver_types_h.cudaError_t with
         Address => Arg1'Address,
         Import;
      Local_Tmp_1 : Interfaces.C.Strings.chars_ptr  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/cuda_runtime_api.h:1259
      := cuda_runtime_api_h.cudaGetErrorString (Local_Tmp_2);
      Local_Tmp_0 : Interfaces.C.Strings.chars_ptr with
         Address => Local_Tmp_1'Address,
         Import;

   begin
      return Local_Tmp_0;
   end Get_Error_String;

   function Get_Device_Count return int is

      Local_Tmp_1 : aliased int;
      Local_Tmp_2 : aliased int with
         Address => Local_Tmp_1'Address,
         Import;
      Local_Tmp_0 : driver_types_h.cudaError_t  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/cuda_runtime_api.h:1288
      := cuda_runtime_api_h.cudaGetDeviceCount (Local_Tmp_1'Unchecked_Access);

   begin
      if Local_Tmp_0 /= 0 then
         Ada.Exceptions.Raise_Exception (CUDA.Exception_Registry.Element (Integer (Local_Tmp_0)));
      end if;
      return Local_Tmp_2;
   end Get_Device_Count;

   function Get_Device_Properties (Device : int) return CUDA.Driver_Types.Device_Prop is

      Local_Tmp_1 : aliased driver_types_h.cudaDeviceProp;
      Local_Tmp_2 : aliased CUDA.Driver_Types.Device_Prop with
         Address => Local_Tmp_1'Address,
         Import;
      Local_Tmp_3 : int with
         Address => Device'Address,
         Import;
      Local_Tmp_0 : driver_types_h.cudaError_t  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/cuda_runtime_api.h:1559
      := cuda_runtime_api_h.cudaGetDeviceProperties (Local_Tmp_1'Unchecked_Access, Local_Tmp_3);

   begin
      if Local_Tmp_0 /= 0 then
         Ada.Exceptions.Raise_Exception (CUDA.Exception_Registry.Element (Integer (Local_Tmp_0)));
      end if;
      return Local_Tmp_2;
   end Get_Device_Properties;

   function Device_Get_Attribute (Attr : CUDA.Driver_Types.Device_Attr; Device : int) return int is

      Local_Tmp_1 : aliased int;
      Local_Tmp_2 : aliased int with
         Address => Local_Tmp_1'Address,
         Import;
      Local_Tmp_3 : driver_types_h.cudaDeviceAttr with
         Address => Attr'Address,
         Import;
      Local_Tmp_4 : int with
         Address => Device'Address,
         Import;
      Local_Tmp_0 : driver_types_h.cudaError_t  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/cuda_runtime_api.h:1748
      := cuda_runtime_api_h.cudaDeviceGetAttribute (Local_Tmp_1'Unchecked_Access, Local_Tmp_3, Local_Tmp_4);

   begin
      if Local_Tmp_0 /= 0 then
         Ada.Exceptions.Raise_Exception (CUDA.Exception_Registry.Element (Integer (Local_Tmp_0)));
      end if;
      return Local_Tmp_2;
   end Device_Get_Attribute;

   procedure Device_Get_Nv_Sci_Sync_Attributes (Nv_Sci_Sync_Attr_List : System.Address; Device : int; Flags : int) is

      Local_Tmp_1 : System.Address with
         Address => Nv_Sci_Sync_Attr_List'Address,
         Import;
      Local_Tmp_2 : int with
         Address => Device'Address,
         Import;
      Local_Tmp_3 : int with
         Address => Flags'Address,
         Import;
      Local_Tmp_0 : driver_types_h.cudaError_t  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/cuda_runtime_api.h:1797
      := cuda_runtime_api_h.cudaDeviceGetNvSciSyncAttributes (Local_Tmp_1, Local_Tmp_2, Local_Tmp_3);

   begin
      if Local_Tmp_0 /= 0 then
         Ada.Exceptions.Raise_Exception (CUDA.Exception_Registry.Element (Integer (Local_Tmp_0)));
      end if;
   end Device_Get_Nv_Sci_Sync_Attributes;

   function Device_Get_P2_PAttribute (Attr : CUDA.Driver_Types.Device_P2_PAttr; Src_Device : int; Dst_Device : int) return int is

      Local_Tmp_1 : aliased int;
      Local_Tmp_2 : aliased int with
         Address => Local_Tmp_1'Address,
         Import;
      Local_Tmp_3 : driver_types_h.cudaDeviceP2PAttr with
         Address => Attr'Address,
         Import;
      Local_Tmp_4 : int with
         Address => Src_Device'Address,
         Import;
      Local_Tmp_5 : int with
         Address => Dst_Device'Address,
         Import;
      Local_Tmp_0 : driver_types_h.cudaError_t  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/cuda_runtime_api.h:1837
      := cuda_runtime_api_h.cudaDeviceGetP2PAttribute (Local_Tmp_1'Unchecked_Access, Local_Tmp_3, Local_Tmp_4, Local_Tmp_5);

   begin
      if Local_Tmp_0 /= 0 then
         Ada.Exceptions.Raise_Exception (CUDA.Exception_Registry.Element (Integer (Local_Tmp_0)));
      end if;
      return Local_Tmp_2;
   end Device_Get_P2_PAttribute;

   procedure Choose_Device (Device : out int; Prop : out CUDA.Driver_Types.Device_Prop) is

      Local_Tmp_1 : aliased int with
         Address => Device'Address,
         Import;
      Local_Tmp_2 : aliased driver_types_h.cudaDeviceProp with
         Address => Prop'Address,
         Import;
      Local_Tmp_0 : driver_types_h.cudaError_t  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/cuda_runtime_api.h:1858
      := cuda_runtime_api_h.cudaChooseDevice (Local_Tmp_1'Access, Local_Tmp_2'Access);

   begin
      if Local_Tmp_0 /= 0 then
         Ada.Exceptions.Raise_Exception (CUDA.Exception_Registry.Element (Integer (Local_Tmp_0)));
      end if;
   end Choose_Device;

   procedure Set_Device (Device : int) is

      Local_Tmp_1 : int with
         Address => Device'Address,
         Import;
      Local_Tmp_0 : driver_types_h.cudaError_t  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/cuda_runtime_api.h:1895
      := cuda_runtime_api_h.cudaSetDevice (Local_Tmp_1);

   begin
      if Local_Tmp_0 /= 0 then
         Ada.Exceptions.Raise_Exception (CUDA.Exception_Registry.Element (Integer (Local_Tmp_0)));
      end if;
   end Set_Device;

   function Get_Device return int is

      Local_Tmp_1 : aliased int;
      Local_Tmp_2 : aliased int with
         Address => Local_Tmp_1'Address,
         Import;
      Local_Tmp_0 : driver_types_h.cudaError_t  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/cuda_runtime_api.h:1916
      := cuda_runtime_api_h.cudaGetDevice (Local_Tmp_1'Unchecked_Access);

   begin
      if Local_Tmp_0 /= 0 then
         Ada.Exceptions.Raise_Exception (CUDA.Exception_Registry.Element (Integer (Local_Tmp_0)));
      end if;
      return Local_Tmp_2;
   end Get_Device;

   procedure Set_Valid_Devices (Device_Arr : out int; Len : int) is

      Local_Tmp_1 : aliased int with
         Address => Device_Arr'Address,
         Import;
      Local_Tmp_2 : int with
         Address => Len'Address,
         Import;
      Local_Tmp_0 : driver_types_h.cudaError_t  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/cuda_runtime_api.h:1947
      := cuda_runtime_api_h.cudaSetValidDevices (Local_Tmp_1'Access, Local_Tmp_2);

   begin
      if Local_Tmp_0 /= 0 then
         Ada.Exceptions.Raise_Exception (CUDA.Exception_Registry.Element (Integer (Local_Tmp_0)));
      end if;
   end Set_Valid_Devices;

   procedure Set_Device_Flags (Flags : unsigned) is

      Local_Tmp_1 : unsigned with
         Address => Flags'Address,
         Import;
      Local_Tmp_0 : driver_types_h.cudaError_t  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/cuda_runtime_api.h:2016
      := cuda_runtime_api_h.cudaSetDeviceFlags (Local_Tmp_1);

   begin
      if Local_Tmp_0 /= 0 then
         Ada.Exceptions.Raise_Exception (CUDA.Exception_Registry.Element (Integer (Local_Tmp_0)));
      end if;
   end Set_Device_Flags;

   function Get_Device_Flags return unsigned is

      Local_Tmp_1 : aliased unsigned;
      Local_Tmp_2 : aliased unsigned with
         Address => Local_Tmp_1'Address,
         Import;
      Local_Tmp_0 : driver_types_h.cudaError_t  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/cuda_runtime_api.h:2062
      := cuda_runtime_api_h.cudaGetDeviceFlags (Local_Tmp_1'Unchecked_Access);

   begin
      if Local_Tmp_0 /= 0 then
         Ada.Exceptions.Raise_Exception (CUDA.Exception_Registry.Element (Integer (Local_Tmp_0)));
      end if;
      return Local_Tmp_2;
   end Get_Device_Flags;

   procedure Stream_Create (P_Stream : System.Address) is

      Local_Tmp_1 : System.Address with
         Address => P_Stream'Address,
         Import;
      Local_Tmp_0 : driver_types_h.cudaError_t  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/cuda_runtime_api.h:2102
      := cuda_runtime_api_h.cudaStreamCreate (Local_Tmp_1);

   begin
      if Local_Tmp_0 /= 0 then
         Ada.Exceptions.Raise_Exception (CUDA.Exception_Registry.Element (Integer (Local_Tmp_0)));
      end if;
   end Stream_Create;

   procedure Stream_Create_With_Flags (P_Stream : System.Address; Flags : unsigned) is

      Local_Tmp_1 : System.Address with
         Address => P_Stream'Address,
         Import;
      Local_Tmp_2 : unsigned with
         Address => Flags'Address,
         Import;
      Local_Tmp_0 : driver_types_h.cudaError_t  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/cuda_runtime_api.h:2134
      := cuda_runtime_api_h.cudaStreamCreateWithFlags (Local_Tmp_1, Local_Tmp_2);

   begin
      if Local_Tmp_0 /= 0 then
         Ada.Exceptions.Raise_Exception (CUDA.Exception_Registry.Element (Integer (Local_Tmp_0)));
      end if;
   end Stream_Create_With_Flags;

   procedure Stream_Create_With_Priority (P_Stream : System.Address; Flags : unsigned; Priority : int) is

      Local_Tmp_1 : System.Address with
         Address => P_Stream'Address,
         Import;
      Local_Tmp_2 : unsigned with
         Address => Flags'Address,
         Import;
      Local_Tmp_3 : int with
         Address => Priority'Address,
         Import;
      Local_Tmp_0 : driver_types_h.cudaError_t  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/cuda_runtime_api.h:2180
      := cuda_runtime_api_h.cudaStreamCreateWithPriority (Local_Tmp_1, Local_Tmp_2, Local_Tmp_3);

   begin
      if Local_Tmp_0 /= 0 then
         Ada.Exceptions.Raise_Exception (CUDA.Exception_Registry.Element (Integer (Local_Tmp_0)));
      end if;
   end Stream_Create_With_Priority;

   procedure Stream_Get_Priority (H_Stream : CUDA.Driver_Types.Stream_T; Priority : out int) is

      Local_Tmp_1 : driver_types_h.cudaStream_t with
         Address => H_Stream'Address,
         Import;
      Local_Tmp_2 : aliased int with
         Address => Priority'Address,
         Import;
      Local_Tmp_0 : driver_types_h.cudaError_t  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/cuda_runtime_api.h:2207
      := cuda_runtime_api_h.cudaStreamGetPriority (Local_Tmp_1, Local_Tmp_2'Access);

   begin
      if Local_Tmp_0 /= 0 then
         Ada.Exceptions.Raise_Exception (CUDA.Exception_Registry.Element (Integer (Local_Tmp_0)));
      end if;
   end Stream_Get_Priority;

   procedure Stream_Get_Flags (H_Stream : CUDA.Driver_Types.Stream_T; Flags : out unsigned) is

      Local_Tmp_1 : driver_types_h.cudaStream_t with
         Address => H_Stream'Address,
         Import;
      Local_Tmp_2 : aliased unsigned with
         Address => Flags'Address,
         Import;
      Local_Tmp_0 : driver_types_h.cudaError_t  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/cuda_runtime_api.h:2232
      := cuda_runtime_api_h.cudaStreamGetFlags (Local_Tmp_1, Local_Tmp_2'Access);

   begin
      if Local_Tmp_0 /= 0 then
         Ada.Exceptions.Raise_Exception (CUDA.Exception_Registry.Element (Integer (Local_Tmp_0)));
      end if;
   end Stream_Get_Flags;

   procedure Stream_Destroy (Stream : CUDA.Driver_Types.Stream_T) is

      Local_Tmp_1 : driver_types_h.cudaStream_t with
         Address => Stream'Address,
         Import;
      Local_Tmp_0 : driver_types_h.cudaError_t  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/cuda_runtime_api.h:2263
      := cuda_runtime_api_h.cudaStreamDestroy (Local_Tmp_1);

   begin
      if Local_Tmp_0 /= 0 then
         Ada.Exceptions.Raise_Exception (CUDA.Exception_Registry.Element (Integer (Local_Tmp_0)));
      end if;
   end Stream_Destroy;

   procedure Stream_Wait_Event (Stream : CUDA.Driver_Types.Stream_T; Event : CUDA.Driver_Types.Event_T; Flags : unsigned) is

      Local_Tmp_1 : driver_types_h.cudaStream_t with
         Address => Stream'Address,
         Import;
      Local_Tmp_2 : driver_types_h.cudaEvent_t with
         Address => Event'Address,
         Import;
      Local_Tmp_3 : unsigned with
         Address => Flags'Address,
         Import;
      Local_Tmp_0 : driver_types_h.cudaError_t  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/cuda_runtime_api.h:2289
      := cuda_runtime_api_h.cudaStreamWaitEvent (Local_Tmp_1, Local_Tmp_2, Local_Tmp_3);

   begin
      if Local_Tmp_0 /= 0 then
         Ada.Exceptions.Raise_Exception (CUDA.Exception_Registry.Element (Integer (Local_Tmp_0)));
      end if;
   end Stream_Wait_Event;

   procedure Stream_Add_Callback (Stream : CUDA.Driver_Types.Stream_T; Callback : Stream_Callback_T; User_Data : System.Address; Flags : unsigned) is

      Local_Tmp_1 : driver_types_h.cudaStream_t with
         Address => Stream'Address,
         Import;
      Local_Tmp_2 : cudaStreamCallback_t with
         Address => Callback'Address,
         Import;
      Local_Tmp_3 : System.Address with
         Address => User_Data'Address,
         Import;
      Local_Tmp_4 : unsigned with
         Address => Flags'Address,
         Import;
      Local_Tmp_0 : driver_types_h.cudaError_t  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/cuda_runtime_api.h:2364
      := cuda_runtime_api_h.cudaStreamAddCallback (Local_Tmp_1, Local_Tmp_2, Local_Tmp_3, Local_Tmp_4);

   begin
      if Local_Tmp_0 /= 0 then
         Ada.Exceptions.Raise_Exception (CUDA.Exception_Registry.Element (Integer (Local_Tmp_0)));
      end if;
   end Stream_Add_Callback;

   procedure Stream_Synchronize (Stream : CUDA.Driver_Types.Stream_T) is

      Local_Tmp_1 : driver_types_h.cudaStream_t with
         Address => Stream'Address,
         Import;
      Local_Tmp_0 : driver_types_h.cudaError_t  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/cuda_runtime_api.h:2388
      := cuda_runtime_api_h.cudaStreamSynchronize (Local_Tmp_1);

   begin
      if Local_Tmp_0 /= 0 then
         Ada.Exceptions.Raise_Exception (CUDA.Exception_Registry.Element (Integer (Local_Tmp_0)));
      end if;
   end Stream_Synchronize;

   procedure Stream_Query (Stream : CUDA.Driver_Types.Stream_T) is

      Local_Tmp_1 : driver_types_h.cudaStream_t with
         Address => Stream'Address,
         Import;
      Local_Tmp_0 : driver_types_h.cudaError_t  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/cuda_runtime_api.h:2413
      := cuda_runtime_api_h.cudaStreamQuery (Local_Tmp_1);

   begin
      if Local_Tmp_0 /= 0 then
         Ada.Exceptions.Raise_Exception (CUDA.Exception_Registry.Element (Integer (Local_Tmp_0)));
      end if;
   end Stream_Query;

   procedure Stream_Attach_Mem_Async (Stream : CUDA.Driver_Types.Stream_T; Dev_Ptr : System.Address; Length : CUDA.Crtdefs.Size_T; Flags : unsigned) is

      Local_Tmp_1 : driver_types_h.cudaStream_t with
         Address => Stream'Address,
         Import;
      Local_Tmp_2 : System.Address with
         Address => Dev_Ptr'Address,
         Import;
      Local_Tmp_3 : crtdefs_h.size_t with
         Address => Length'Address,
         Import;
      Local_Tmp_4 : unsigned with
         Address => Flags'Address,
         Import;
      Local_Tmp_0 : driver_types_h.cudaError_t  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/cuda_runtime_api.h:2496
      := cuda_runtime_api_h.cudaStreamAttachMemAsync (Local_Tmp_1, Local_Tmp_2, Local_Tmp_3, Local_Tmp_4);

   begin
      if Local_Tmp_0 /= 0 then
         Ada.Exceptions.Raise_Exception (CUDA.Exception_Registry.Element (Integer (Local_Tmp_0)));
      end if;
   end Stream_Attach_Mem_Async;

   procedure Stream_Begin_Capture (Stream : CUDA.Driver_Types.Stream_T; Mode : CUDA.Driver_Types.Stream_Capture_Mode) is

      Local_Tmp_1 : driver_types_h.cudaStream_t with
         Address => Stream'Address,
         Import;
      Local_Tmp_2 : driver_types_h.cudaStreamCaptureMode with
         Address => Mode'Address,
         Import;
      Local_Tmp_0 : driver_types_h.cudaError_t  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/cuda_runtime_api.h:2532
      := cuda_runtime_api_h.cudaStreamBeginCapture (Local_Tmp_1, Local_Tmp_2);

   begin
      if Local_Tmp_0 /= 0 then
         Ada.Exceptions.Raise_Exception (CUDA.Exception_Registry.Element (Integer (Local_Tmp_0)));
      end if;
   end Stream_Begin_Capture;

   procedure Thread_Exchange_Stream_Capture_Mode (Mode : out CUDA.Driver_Types.Stream_Capture_Mode) is

      Local_Tmp_1 : aliased driver_types_h.cudaStreamCaptureMode with
         Address => Mode'Address,
         Import;
      Local_Tmp_0 : driver_types_h.cudaError_t  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/cuda_runtime_api.h:2583
      := cuda_runtime_api_h.cudaThreadExchangeStreamCaptureMode (Local_Tmp_1'Access);

   begin
      if Local_Tmp_0 /= 0 then
         Ada.Exceptions.Raise_Exception (CUDA.Exception_Registry.Element (Integer (Local_Tmp_0)));
      end if;
   end Thread_Exchange_Stream_Capture_Mode;

   procedure Stream_End_Capture (Stream : CUDA.Driver_Types.Stream_T; P_Graph : System.Address) is

      Local_Tmp_1 : driver_types_h.cudaStream_t with
         Address => Stream'Address,
         Import;
      Local_Tmp_2 : System.Address with
         Address => P_Graph'Address,
         Import;
      Local_Tmp_0 : driver_types_h.cudaError_t  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/cuda_runtime_api.h:2611
      := cuda_runtime_api_h.cudaStreamEndCapture (Local_Tmp_1, Local_Tmp_2);

   begin
      if Local_Tmp_0 /= 0 then
         Ada.Exceptions.Raise_Exception (CUDA.Exception_Registry.Element (Integer (Local_Tmp_0)));
      end if;
   end Stream_End_Capture;

   procedure Stream_Is_Capturing (Stream : CUDA.Driver_Types.Stream_T; P_Capture_Status : out CUDA.Driver_Types.Stream_Capture_Status) is

      Local_Tmp_1 : driver_types_h.cudaStream_t with
         Address => Stream'Address,
         Import;
      Local_Tmp_2 : aliased driver_types_h.cudaStreamCaptureStatus with
         Address => P_Capture_Status'Address,
         Import;
      Local_Tmp_0 : driver_types_h.cudaError_t  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/cuda_runtime_api.h:2649
      := cuda_runtime_api_h.cudaStreamIsCapturing (Local_Tmp_1, Local_Tmp_2'Access);

   begin
      if Local_Tmp_0 /= 0 then
         Ada.Exceptions.Raise_Exception (CUDA.Exception_Registry.Element (Integer (Local_Tmp_0)));
      end if;
   end Stream_Is_Capturing;

   procedure Stream_Get_Capture_Info (Stream : CUDA.Driver_Types.Stream_T; P_Capture_Status : out CUDA.Driver_Types.Stream_Capture_Status; P_Id : out Extensions.unsigned_long_long) is

      Local_Tmp_1 : driver_types_h.cudaStream_t with
         Address => Stream'Address,
         Import;
      Local_Tmp_2 : aliased driver_types_h.cudaStreamCaptureStatus with
         Address => P_Capture_Status'Address,
         Import;
      Local_Tmp_3 : aliased Extensions.unsigned_long_long with
         Address => P_Id'Address,
         Import;
      Local_Tmp_0 : driver_types_h.cudaError_t  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/cuda_runtime_api.h:2677
      := cuda_runtime_api_h.cudaStreamGetCaptureInfo (Local_Tmp_1, Local_Tmp_2'Access, Local_Tmp_3'Access);

   begin
      if Local_Tmp_0 /= 0 then
         Ada.Exceptions.Raise_Exception (CUDA.Exception_Registry.Element (Integer (Local_Tmp_0)));
      end if;
   end Stream_Get_Capture_Info;

   function Event_Create return CUDA.Driver_Types.Event_T is

      Local_Tmp_1 : aliased CUDA.Driver_Types.Event_T;
      Local_Tmp_0 : driver_types_h.cudaError_t  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/cuda_runtime_api.h:2714
      := cuda_runtime_api_h.cudaEventCreate (Local_Tmp_1'Address);

   begin
      if Local_Tmp_0 /= 0 then
         Ada.Exceptions.Raise_Exception (CUDA.Exception_Registry.Element (Integer (Local_Tmp_0)));
      end if;
      return Local_Tmp_1;
   end Event_Create;

   function Event_Create_With_Flags (Flags : unsigned) return CUDA.Driver_Types.Event_T is

      Local_Tmp_1 : aliased CUDA.Driver_Types.Event_T;
      Local_Tmp_2 : unsigned with
         Address => Flags'Address,
         Import;
      Local_Tmp_0 : driver_types_h.cudaError_t  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/cuda_runtime_api.h:2751
      := cuda_runtime_api_h.cudaEventCreateWithFlags (Local_Tmp_1'Address, Local_Tmp_2);

   begin
      if Local_Tmp_0 /= 0 then
         Ada.Exceptions.Raise_Exception (CUDA.Exception_Registry.Element (Integer (Local_Tmp_0)));
      end if;
      return Local_Tmp_1;
   end Event_Create_With_Flags;

   procedure Event_Record (Event : CUDA.Driver_Types.Event_T; Stream : CUDA.Driver_Types.Stream_T) is

      Local_Tmp_1 : driver_types_h.cudaEvent_t with
         Address => Event'Address,
         Import;
      Local_Tmp_2 : driver_types_h.cudaStream_t with
         Address => Stream'Address,
         Import;
      Local_Tmp_0 : driver_types_h.cudaError_t  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/cuda_runtime_api.h:2790
      := cuda_runtime_api_h.cudaEventRecord (Local_Tmp_1, Local_Tmp_2);

   begin
      if Local_Tmp_0 /= 0 then
         Ada.Exceptions.Raise_Exception (CUDA.Exception_Registry.Element (Integer (Local_Tmp_0)));
      end if;
   end Event_Record;

   procedure Event_Query (Event : CUDA.Driver_Types.Event_T) is

      Local_Tmp_1 : driver_types_h.cudaEvent_t with
         Address => Event'Address,
         Import;
      Local_Tmp_0 : driver_types_h.cudaError_t  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/cuda_runtime_api.h:2821
      := cuda_runtime_api_h.cudaEventQuery (Local_Tmp_1);

   begin
      if Local_Tmp_0 /= 0 then
         Ada.Exceptions.Raise_Exception (CUDA.Exception_Registry.Element (Integer (Local_Tmp_0)));
      end if;
   end Event_Query;

   procedure Event_Synchronize (Event : CUDA.Driver_Types.Event_T) is

      Local_Tmp_1 : driver_types_h.cudaEvent_t with
         Address => Event'Address,
         Import;
      Local_Tmp_0 : driver_types_h.cudaError_t  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/cuda_runtime_api.h:2851
      := cuda_runtime_api_h.cudaEventSynchronize (Local_Tmp_1);

   begin
      if Local_Tmp_0 /= 0 then
         Ada.Exceptions.Raise_Exception (CUDA.Exception_Registry.Element (Integer (Local_Tmp_0)));
      end if;
   end Event_Synchronize;

   procedure Event_Destroy (Event : CUDA.Driver_Types.Event_T) is

      Local_Tmp_1 : driver_types_h.cudaEvent_t with
         Address => Event'Address,
         Import;
      Local_Tmp_0 : driver_types_h.cudaError_t  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/cuda_runtime_api.h:2878
      := cuda_runtime_api_h.cudaEventDestroy (Local_Tmp_1);

   begin
      if Local_Tmp_0 /= 0 then
         Ada.Exceptions.Raise_Exception (CUDA.Exception_Registry.Element (Integer (Local_Tmp_0)));
      end if;
   end Event_Destroy;

   procedure Event_Elapsed_Time (Ms : out Float; Start : CUDA.Driver_Types.Event_T; C_End : CUDA.Driver_Types.Event_T) is

      Local_Tmp_1 : aliased Float with
         Address => Ms'Address,
         Import;
      Local_Tmp_2 : driver_types_h.cudaEvent_t with
         Address => Start'Address,
         Import;
      Local_Tmp_3 : driver_types_h.cudaEvent_t with
         Address => C_End'Address,
         Import;
      Local_Tmp_0 : driver_types_h.cudaError_t  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/cuda_runtime_api.h:2921
      := cuda_runtime_api_h.cudaEventElapsedTime (Local_Tmp_1'Access, Local_Tmp_2, Local_Tmp_3);

   begin
      if Local_Tmp_0 /= 0 then
         Ada.Exceptions.Raise_Exception (CUDA.Exception_Registry.Element (Integer (Local_Tmp_0)));
      end if;
   end Event_Elapsed_Time;

   procedure Import_External_Memory (Ext_Mem_Out : System.Address; Mem_Handle_Desc : out CUDA.Driver_Types.External_Memory_Handle_Desc) is

      Local_Tmp_1 : System.Address with
         Address => Ext_Mem_Out'Address,
         Import;
      Local_Tmp_2 : aliased driver_types_h.cudaExternalMemoryHandleDesc with
         Address => Mem_Handle_Desc'Address,
         Import;
      Local_Tmp_0 : driver_types_h.cudaError_t  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/cuda_runtime_api.h:3098
      := cuda_runtime_api_h.cudaImportExternalMemory (Local_Tmp_1, Local_Tmp_2'Access);

   begin
      if Local_Tmp_0 /= 0 then
         Ada.Exceptions.Raise_Exception (CUDA.Exception_Registry.Element (Integer (Local_Tmp_0)));
      end if;
   end Import_External_Memory;

   procedure External_Memory_Get_Mapped_Buffer (Dev_Ptr : System.Address; Ext_Mem : CUDA.Driver_Types.External_Memory_T; Buffer_Desc : out CUDA.Driver_Types.External_Memory_Buffer_Desc) is

      Local_Tmp_1 : System.Address with
         Address => Dev_Ptr'Address,
         Import;
      Local_Tmp_2 : driver_types_h.cudaExternalMemory_t with
         Address => Ext_Mem'Address,
         Import;
      Local_Tmp_3 : aliased driver_types_h.cudaExternalMemoryBufferDesc with
         Address => Buffer_Desc'Address,
         Import;
      Local_Tmp_0 : driver_types_h.cudaError_t  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/cuda_runtime_api.h:3152
      := cuda_runtime_api_h.cudaExternalMemoryGetMappedBuffer (Local_Tmp_1, Local_Tmp_2, Local_Tmp_3'Access);

   begin
      if Local_Tmp_0 /= 0 then
         Ada.Exceptions.Raise_Exception (CUDA.Exception_Registry.Element (Integer (Local_Tmp_0)));
      end if;
   end External_Memory_Get_Mapped_Buffer;

   procedure External_Memory_Get_Mapped_Mipmapped_Array (Mipmap : System.Address; Ext_Mem : CUDA.Driver_Types.External_Memory_T; Mipmap_Desc : out CUDA.Driver_Types.External_Memory_Mipmapped_Array_Desc) is

      Local_Tmp_1 : System.Address with
         Address => Mipmap'Address,
         Import;
      Local_Tmp_2 : driver_types_h.cudaExternalMemory_t with
         Address => Ext_Mem'Address,
         Import;
      Local_Tmp_3 : aliased driver_types_h.cudaExternalMemoryMipmappedArrayDesc with
         Address => Mipmap_Desc'Address,
         Import;
      Local_Tmp_0 : driver_types_h.cudaError_t  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/cuda_runtime_api.h:3211
      := cuda_runtime_api_h.cudaExternalMemoryGetMappedMipmappedArray (Local_Tmp_1, Local_Tmp_2, Local_Tmp_3'Access);

   begin
      if Local_Tmp_0 /= 0 then
         Ada.Exceptions.Raise_Exception (CUDA.Exception_Registry.Element (Integer (Local_Tmp_0)));
      end if;
   end External_Memory_Get_Mapped_Mipmapped_Array;

   procedure Destroy_External_Memory (Ext_Mem : CUDA.Driver_Types.External_Memory_T) is

      Local_Tmp_1 : driver_types_h.cudaExternalMemory_t with
         Address => Ext_Mem'Address,
         Import;
      Local_Tmp_0 : driver_types_h.cudaError_t  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/cuda_runtime_api.h:3234
      := cuda_runtime_api_h.cudaDestroyExternalMemory (Local_Tmp_1);

   begin
      if Local_Tmp_0 /= 0 then
         Ada.Exceptions.Raise_Exception (CUDA.Exception_Registry.Element (Integer (Local_Tmp_0)));
      end if;
   end Destroy_External_Memory;

   procedure Import_External_Semaphore (Ext_Sem_Out : System.Address; Sem_Handle_Desc : out CUDA.Driver_Types.External_Semaphore_Handle_Desc) is

      Local_Tmp_1 : System.Address with
         Address => Ext_Sem_Out'Address,
         Import;
      Local_Tmp_2 : aliased driver_types_h.cudaExternalSemaphoreHandleDesc with
         Address => Sem_Handle_Desc'Address,
         Import;
      Local_Tmp_0 : driver_types_h.cudaError_t  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/cuda_runtime_api.h:3365
      := cuda_runtime_api_h.cudaImportExternalSemaphore (Local_Tmp_1, Local_Tmp_2'Access);

   begin
      if Local_Tmp_0 /= 0 then
         Ada.Exceptions.Raise_Exception (CUDA.Exception_Registry.Element (Integer (Local_Tmp_0)));
      end if;
   end Import_External_Semaphore;

   procedure Signal_External_Semaphores_Async (Ext_Sem_Array : System.Address; Params_Array : out CUDA.Driver_Types.External_Semaphore_Signal_Params; Num_Ext_Sems : unsigned; Stream : CUDA.Driver_Types.Stream_T) is

      Local_Tmp_1 : System.Address with
         Address => Ext_Sem_Array'Address,
         Import;
      Local_Tmp_2 : aliased driver_types_h.cudaExternalSemaphoreSignalParams with
         Address => Params_Array'Address,
         Import;
      Local_Tmp_3 : unsigned with
         Address => Num_Ext_Sems'Address,
         Import;
      Local_Tmp_4 : driver_types_h.cudaStream_t with
         Address => Stream'Address,
         Import;
      Local_Tmp_0 : driver_types_h.cudaError_t  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/cuda_runtime_api.h:3430
      := cuda_runtime_api_h.cudaSignalExternalSemaphoresAsync (Local_Tmp_1, Local_Tmp_2'Access, Local_Tmp_3, Local_Tmp_4);

   begin
      if Local_Tmp_0 /= 0 then
         Ada.Exceptions.Raise_Exception (CUDA.Exception_Registry.Element (Integer (Local_Tmp_0)));
      end if;
   end Signal_External_Semaphores_Async;

   procedure Wait_External_Semaphores_Async (Ext_Sem_Array : System.Address; Params_Array : out CUDA.Driver_Types.External_Semaphore_Wait_Params; Num_Ext_Sems : unsigned; Stream : CUDA.Driver_Types.Stream_T) is

      Local_Tmp_1 : System.Address with
         Address => Ext_Sem_Array'Address,
         Import;
      Local_Tmp_2 : aliased driver_types_h.cudaExternalSemaphoreWaitParams with
         Address => Params_Array'Address,
         Import;
      Local_Tmp_3 : unsigned with
         Address => Num_Ext_Sems'Address,
         Import;
      Local_Tmp_4 : driver_types_h.cudaStream_t with
         Address => Stream'Address,
         Import;
      Local_Tmp_0 : driver_types_h.cudaError_t  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/cuda_runtime_api.h:3504
      := cuda_runtime_api_h.cudaWaitExternalSemaphoresAsync (Local_Tmp_1, Local_Tmp_2'Access, Local_Tmp_3, Local_Tmp_4);

   begin
      if Local_Tmp_0 /= 0 then
         Ada.Exceptions.Raise_Exception (CUDA.Exception_Registry.Element (Integer (Local_Tmp_0)));
      end if;
   end Wait_External_Semaphores_Async;

   procedure Destroy_External_Semaphore (Ext_Sem : CUDA.Driver_Types.External_Semaphore_T) is

      Local_Tmp_1 : driver_types_h.cudaExternalSemaphore_t with
         Address => Ext_Sem'Address,
         Import;
      Local_Tmp_0 : driver_types_h.cudaError_t  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/cuda_runtime_api.h:3526
      := cuda_runtime_api_h.cudaDestroyExternalSemaphore (Local_Tmp_1);

   begin
      if Local_Tmp_0 /= 0 then
         Ada.Exceptions.Raise_Exception (CUDA.Exception_Registry.Element (Integer (Local_Tmp_0)));
      end if;
   end Destroy_External_Semaphore;

   procedure Launch_Kernel (Func : System.Address; Grid_Dim : CUDA.Vector_Types.Dim3; Block_Dim : CUDA.Vector_Types.Dim3; Args : System.Address; Shared_Mem : CUDA.Crtdefs.Size_T; Stream : CUDA.Driver_Types.Stream_T) is

      Local_Tmp_1 : System.Address with
         Address => Func'Address,
         Import;
      Local_Tmp_2 : vector_types_h.dim3 with
         Address => Grid_Dim'Address,
         Import;
      Local_Tmp_3 : vector_types_h.dim3 with
         Address => Block_Dim'Address,
         Import;
      Local_Tmp_4 : System.Address with
         Address => Args'Address,
         Import;
      Local_Tmp_5 : crtdefs_h.size_t with
         Address => Shared_Mem'Address,
         Import;
      Local_Tmp_6 : driver_types_h.cudaStream_t with
         Address => Stream'Address,
         Import;
      Local_Tmp_0 : driver_types_h.cudaError_t  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/cuda_runtime_api.h:3591
      := cuda_runtime_api_h.cudaLaunchKernel (Local_Tmp_1, Local_Tmp_2, Local_Tmp_3, Local_Tmp_4, Local_Tmp_5, Local_Tmp_6);

   begin
      if Local_Tmp_0 /= 0 then
         Ada.Exceptions.Raise_Exception (CUDA.Exception_Registry.Element (Integer (Local_Tmp_0)));
      end if;
   end Launch_Kernel;

   procedure Launch_Cooperative_Kernel (Func : System.Address; Grid_Dim : CUDA.Vector_Types.Dim3; Block_Dim : CUDA.Vector_Types.Dim3; Args : System.Address; Shared_Mem : CUDA.Crtdefs.Size_T; Stream : CUDA.Driver_Types.Stream_T) is

      Local_Tmp_1 : System.Address with
         Address => Func'Address,
         Import;
      Local_Tmp_2 : vector_types_h.dim3 with
         Address => Grid_Dim'Address,
         Import;
      Local_Tmp_3 : vector_types_h.dim3 with
         Address => Block_Dim'Address,
         Import;
      Local_Tmp_4 : System.Address with
         Address => Args'Address,
         Import;
      Local_Tmp_5 : crtdefs_h.size_t with
         Address => Shared_Mem'Address,
         Import;
      Local_Tmp_6 : driver_types_h.cudaStream_t with
         Address => Stream'Address,
         Import;
      Local_Tmp_0 : driver_types_h.cudaError_t  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/cuda_runtime_api.h:3648
      := cuda_runtime_api_h.cudaLaunchCooperativeKernel (Local_Tmp_1, Local_Tmp_2, Local_Tmp_3, Local_Tmp_4, Local_Tmp_5, Local_Tmp_6);

   begin
      if Local_Tmp_0 /= 0 then
         Ada.Exceptions.Raise_Exception (CUDA.Exception_Registry.Element (Integer (Local_Tmp_0)));
      end if;
   end Launch_Cooperative_Kernel;

   procedure Launch_Cooperative_Kernel_Multi_Device (Launch_Params_List : out CUDA.Driver_Types.Launch_Params; Num_Devices : unsigned; Flags : unsigned) is

      Local_Tmp_1 : aliased driver_types_h.cudaLaunchParams with
         Address => Launch_Params_List'Address,
         Import;
      Local_Tmp_2 : unsigned with
         Address => Num_Devices'Address,
         Import;
      Local_Tmp_3 : unsigned with
         Address => Flags'Address,
         Import;
      Local_Tmp_0 : driver_types_h.cudaError_t  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/cuda_runtime_api.h:3747
      := cuda_runtime_api_h.cudaLaunchCooperativeKernelMultiDevice (Local_Tmp_1'Access, Local_Tmp_2, Local_Tmp_3);

   begin
      if Local_Tmp_0 /= 0 then
         Ada.Exceptions.Raise_Exception (CUDA.Exception_Registry.Element (Integer (Local_Tmp_0)));
      end if;
   end Launch_Cooperative_Kernel_Multi_Device;

   procedure Func_Set_Cache_Config (Func : System.Address; Cache_Config : CUDA.Driver_Types.Func_Cache) is

      Local_Tmp_1 : System.Address with
         Address => Func'Address,
         Import;
      Local_Tmp_2 : driver_types_h.cudaFuncCache with
         Address => Cache_Config'Address,
         Import;
      Local_Tmp_0 : driver_types_h.cudaError_t  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/cuda_runtime_api.h:3796
      := cuda_runtime_api_h.cudaFuncSetCacheConfig (Local_Tmp_1, Local_Tmp_2);

   begin
      if Local_Tmp_0 /= 0 then
         Ada.Exceptions.Raise_Exception (CUDA.Exception_Registry.Element (Integer (Local_Tmp_0)));
      end if;
   end Func_Set_Cache_Config;

   procedure Func_Set_Shared_Mem_Config (Func : System.Address; Config : CUDA.Driver_Types.Shared_Mem_Config) is

      Local_Tmp_1 : System.Address with
         Address => Func'Address,
         Import;
      Local_Tmp_2 : driver_types_h.cudaSharedMemConfig with
         Address => Config'Address,
         Import;
      Local_Tmp_0 : driver_types_h.cudaError_t  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/cuda_runtime_api.h:3851
      := cuda_runtime_api_h.cudaFuncSetSharedMemConfig (Local_Tmp_1, Local_Tmp_2);

   begin
      if Local_Tmp_0 /= 0 then
         Ada.Exceptions.Raise_Exception (CUDA.Exception_Registry.Element (Integer (Local_Tmp_0)));
      end if;
   end Func_Set_Shared_Mem_Config;

   function Func_Get_Attributes (Func : System.Address) return CUDA.Driver_Types.Func_Attributes is

      Local_Tmp_1 : aliased driver_types_h.cudaFuncAttributes;
      Local_Tmp_2 : aliased CUDA.Driver_Types.Func_Attributes with
         Address => Local_Tmp_1'Address,
         Import;
      Local_Tmp_3 : System.Address with
         Address => Func'Address,
         Import;
      Local_Tmp_0 : driver_types_h.cudaError_t  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/cuda_runtime_api.h:3886
      := cuda_runtime_api_h.cudaFuncGetAttributes (Local_Tmp_1'Unchecked_Access, Local_Tmp_3);

   begin
      if Local_Tmp_0 /= 0 then
         Ada.Exceptions.Raise_Exception (CUDA.Exception_Registry.Element (Integer (Local_Tmp_0)));
      end if;
      return Local_Tmp_2;
   end Func_Get_Attributes;

   procedure Func_Set_Attribute (Func : System.Address; Attr : CUDA.Driver_Types.Func_Attribute; Value : int) is

      Local_Tmp_1 : System.Address with
         Address => Func'Address,
         Import;
      Local_Tmp_2 : driver_types_h.cudaFuncAttribute with
         Address => Attr'Address,
         Import;
      Local_Tmp_3 : int with
         Address => Value'Address,
         Import;
      Local_Tmp_0 : driver_types_h.cudaError_t  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/cuda_runtime_api.h:3925
      := cuda_runtime_api_h.cudaFuncSetAttribute (Local_Tmp_1, Local_Tmp_2, Local_Tmp_3);

   begin
      if Local_Tmp_0 /= 0 then
         Ada.Exceptions.Raise_Exception (CUDA.Exception_Registry.Element (Integer (Local_Tmp_0)));
      end if;
   end Func_Set_Attribute;

   procedure Set_Double_For_Device (D : out double) is

      Local_Tmp_1 : aliased double with
         Address => D'Address,
         Import;
      Local_Tmp_0 : driver_types_h.cudaError_t  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/cuda_runtime_api.h:3949
      := cuda_runtime_api_h.cudaSetDoubleForDevice (Local_Tmp_1'Access);

   begin
      if Local_Tmp_0 /= 0 then
         Ada.Exceptions.Raise_Exception (CUDA.Exception_Registry.Element (Integer (Local_Tmp_0)));
      end if;
   end Set_Double_For_Device;

   procedure Set_Double_For_Host (D : out double) is

      Local_Tmp_1 : aliased double with
         Address => D'Address,
         Import;
      Local_Tmp_0 : driver_types_h.cudaError_t  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/cuda_runtime_api.h:3973
      := cuda_runtime_api_h.cudaSetDoubleForHost (Local_Tmp_1'Access);

   begin
      if Local_Tmp_0 /= 0 then
         Ada.Exceptions.Raise_Exception (CUDA.Exception_Registry.Element (Integer (Local_Tmp_0)));
      end if;
   end Set_Double_For_Host;

   procedure Launch_Host_Func (Stream : CUDA.Driver_Types.Stream_T; Fn : CUDA.Driver_Types.Host_Fn_T; User_Data : System.Address) is

      Local_Tmp_1 : driver_types_h.cudaStream_t with
         Address => Stream'Address,
         Import;
      Local_Tmp_2 : driver_types_h.cudaHostFn_t with
         Address => Fn'Address,
         Import;
      Local_Tmp_3 : System.Address with
         Address => User_Data'Address,
         Import;
      Local_Tmp_0 : driver_types_h.cudaError_t  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/cuda_runtime_api.h:4039
      := cuda_runtime_api_h.cudaLaunchHostFunc (Local_Tmp_1, Local_Tmp_2, Local_Tmp_3);

   begin
      if Local_Tmp_0 /= 0 then
         Ada.Exceptions.Raise_Exception (CUDA.Exception_Registry.Element (Integer (Local_Tmp_0)));
      end if;
   end Launch_Host_Func;

   procedure Occupancy_Max_Active_Blocks_Per_Multiprocessor (Num_Blocks : out int; Func : System.Address; Block_Size : int; Dynamic_SMem_Size : CUDA.Crtdefs.Size_T) is

      Local_Tmp_1 : aliased int with
         Address => Num_Blocks'Address,
         Import;
      Local_Tmp_2 : System.Address with
         Address => Func'Address,
         Import;
      Local_Tmp_3 : int with
         Address => Block_Size'Address,
         Import;
      Local_Tmp_4 : crtdefs_h.size_t with
         Address => Dynamic_SMem_Size'Address,
         Import;
      Local_Tmp_0 : driver_types_h.cudaError_t  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/cuda_runtime_api.h:4094
      := cuda_runtime_api_h.cudaOccupancyMaxActiveBlocksPerMultiprocessor (Local_Tmp_1'Access, Local_Tmp_2, Local_Tmp_3, Local_Tmp_4);

   begin
      if Local_Tmp_0 /= 0 then
         Ada.Exceptions.Raise_Exception (CUDA.Exception_Registry.Element (Integer (Local_Tmp_0)));
      end if;
   end Occupancy_Max_Active_Blocks_Per_Multiprocessor;

   procedure Occupancy_Max_Active_Blocks_Per_Multiprocessor_With_Flags (Num_Blocks : out int; Func : System.Address; Block_Size : int; Dynamic_SMem_Size : CUDA.Crtdefs.Size_T; Flags : unsigned) is

      Local_Tmp_1 : aliased int with
         Address => Num_Blocks'Address,
         Import;
      Local_Tmp_2 : System.Address with
         Address => Func'Address,
         Import;
      Local_Tmp_3 : int with
         Address => Block_Size'Address,
         Import;
      Local_Tmp_4 : crtdefs_h.size_t with
         Address => Dynamic_SMem_Size'Address,
         Import;
      Local_Tmp_5 : unsigned with
         Address => Flags'Address,
         Import;
      Local_Tmp_0 : driver_types_h.cudaError_t  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/cuda_runtime_api.h:4138
      := cuda_runtime_api_h.cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags (Local_Tmp_1'Access, Local_Tmp_2, Local_Tmp_3, Local_Tmp_4, Local_Tmp_5);

   begin
      if Local_Tmp_0 /= 0 then
         Ada.Exceptions.Raise_Exception (CUDA.Exception_Registry.Element (Integer (Local_Tmp_0)));
      end if;
   end Occupancy_Max_Active_Blocks_Per_Multiprocessor_With_Flags;

   procedure Malloc_Managed (Dev_Ptr : System.Address; Size : CUDA.Crtdefs.Size_T; Flags : unsigned) is

      Local_Tmp_1 : System.Address with
         Address => Dev_Ptr'Address,
         Import;
      Local_Tmp_2 : crtdefs_h.size_t with
         Address => Size'Address,
         Import;
      Local_Tmp_3 : unsigned with
         Address => Flags'Address,
         Import;
      Local_Tmp_0 : driver_types_h.cudaError_t  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/cuda_runtime_api.h:4258
      := cuda_runtime_api_h.cudaMallocManaged (Local_Tmp_1, Local_Tmp_2, Local_Tmp_3);

   begin
      if Local_Tmp_0 /= 0 then
         Ada.Exceptions.Raise_Exception (CUDA.Exception_Registry.Element (Integer (Local_Tmp_0)));
      end if;
   end Malloc_Managed;

   function Malloc (Size : CUDA.Crtdefs.Size_T) return System.Address is

      Tmp         : aliased System.Address;
      Local_Tmp_1 : crtdefs_h.size_t with
         Address => Size'Address,
         Import;
      Local_Tmp_0 : driver_types_h.cudaError_t  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/cuda_runtime_api.h:4289
      := cuda_runtime_api_h.cudaMalloc (Tmp'Address, Local_Tmp_1);

   begin
      if Local_Tmp_0 /= 0 then
         Ada.Exceptions.Raise_Exception (CUDA.Exception_Registry.Element (Integer (Local_Tmp_0)));
      end if;
      return Tmp;
   end Malloc;

   procedure Malloc_Host (Ptr : System.Address; Size : CUDA.Crtdefs.Size_T) is

      Local_Tmp_1 : System.Address with
         Address => Ptr'Address,
         Import;
      Local_Tmp_2 : crtdefs_h.size_t with
         Address => Size'Address,
         Import;
      Local_Tmp_0 : driver_types_h.cudaError_t  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/cuda_runtime_api.h:4322
      := cuda_runtime_api_h.cudaMallocHost (Local_Tmp_1, Local_Tmp_2);

   begin
      if Local_Tmp_0 /= 0 then
         Ada.Exceptions.Raise_Exception (CUDA.Exception_Registry.Element (Integer (Local_Tmp_0)));
      end if;
   end Malloc_Host;

   procedure Malloc_Pitch (Dev_Ptr : System.Address; Pitch : out CUDA.Crtdefs.Size_T; Width : CUDA.Crtdefs.Size_T; Height : CUDA.Crtdefs.Size_T) is

      Local_Tmp_1 : System.Address with
         Address => Dev_Ptr'Address,
         Import;
      Local_Tmp_2 : aliased crtdefs_h.size_t with
         Address => Pitch'Address,
         Import;
      Local_Tmp_3 : crtdefs_h.size_t with
         Address => Width'Address,
         Import;
      Local_Tmp_4 : crtdefs_h.size_t with
         Address => Height'Address,
         Import;
      Local_Tmp_0 : driver_types_h.cudaError_t  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/cuda_runtime_api.h:4365
      := cuda_runtime_api_h.cudaMallocPitch (Local_Tmp_1, Local_Tmp_2'Access, Local_Tmp_3, Local_Tmp_4);

   begin
      if Local_Tmp_0 /= 0 then
         Ada.Exceptions.Raise_Exception (CUDA.Exception_Registry.Element (Integer (Local_Tmp_0)));
      end if;
   end Malloc_Pitch;

   procedure Malloc_Array (C_Array : System.Address; Desc : out CUDA.Driver_Types.Channel_Format_Desc; Width : CUDA.Crtdefs.Size_T; Height : CUDA.Crtdefs.Size_T; Flags : unsigned) is

      Local_Tmp_1 : System.Address with
         Address => C_Array'Address,
         Import;
      Local_Tmp_2 : aliased driver_types_h.cudaChannelFormatDesc with
         Address => Desc'Address,
         Import;
      Local_Tmp_3 : crtdefs_h.size_t with
         Address => Width'Address,
         Import;
      Local_Tmp_4 : crtdefs_h.size_t with
         Address => Height'Address,
         Import;
      Local_Tmp_5 : unsigned with
         Address => Flags'Address,
         Import;
      Local_Tmp_0 : driver_types_h.cudaError_t  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/cuda_runtime_api.h:4411
      := cuda_runtime_api_h.cudaMallocArray (Local_Tmp_1, Local_Tmp_2'Access, Local_Tmp_3, Local_Tmp_4, Local_Tmp_5);

   begin
      if Local_Tmp_0 /= 0 then
         Ada.Exceptions.Raise_Exception (CUDA.Exception_Registry.Element (Integer (Local_Tmp_0)));
      end if;
   end Malloc_Array;

   procedure Free (Dev_Ptr : System.Address) is

      Local_Tmp_1 : System.Address with
         Address => Dev_Ptr'Address,
         Import;
      Local_Tmp_0 : driver_types_h.cudaError_t  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/cuda_runtime_api.h:4440
      := cuda_runtime_api_h.cudaFree (Local_Tmp_1);

   begin
      if Local_Tmp_0 /= 0 then
         Ada.Exceptions.Raise_Exception (CUDA.Exception_Registry.Element (Integer (Local_Tmp_0)));
      end if;
   end Free;

   procedure Free_Host (Ptr : System.Address) is

      Local_Tmp_1 : System.Address with
         Address => Ptr'Address,
         Import;
      Local_Tmp_0 : driver_types_h.cudaError_t  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/cuda_runtime_api.h:4463
      := cuda_runtime_api_h.cudaFreeHost (Local_Tmp_1);

   begin
      if Local_Tmp_0 /= 0 then
         Ada.Exceptions.Raise_Exception (CUDA.Exception_Registry.Element (Integer (Local_Tmp_0)));
      end if;
   end Free_Host;

   procedure Free_Array (C_Array : CUDA.Driver_Types.CUDA_Array_T) is

      Local_Tmp_1 : driver_types_h.cudaArray_t with
         Address => C_Array'Address,
         Import;
      Local_Tmp_0 : driver_types_h.cudaError_t  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/cuda_runtime_api.h:4486
      := cuda_runtime_api_h.cudaFreeArray (Local_Tmp_1);

   begin
      if Local_Tmp_0 /= 0 then
         Ada.Exceptions.Raise_Exception (CUDA.Exception_Registry.Element (Integer (Local_Tmp_0)));
      end if;
   end Free_Array;

   procedure Free_Mipmapped_Array (Mipmapped_Array : CUDA.Driver_Types.Mipmapped_Array_T) is

      Local_Tmp_1 : driver_types_h.cudaMipmappedArray_t with
         Address => Mipmapped_Array'Address,
         Import;
      Local_Tmp_0 : driver_types_h.cudaError_t  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/cuda_runtime_api.h:4509
      := cuda_runtime_api_h.cudaFreeMipmappedArray (Local_Tmp_1);

   begin
      if Local_Tmp_0 /= 0 then
         Ada.Exceptions.Raise_Exception (CUDA.Exception_Registry.Element (Integer (Local_Tmp_0)));
      end if;
   end Free_Mipmapped_Array;

   procedure Host_Alloc (P_Host : System.Address; Size : CUDA.Crtdefs.Size_T; Flags : unsigned) is

      Local_Tmp_1 : System.Address with
         Address => P_Host'Address,
         Import;
      Local_Tmp_2 : crtdefs_h.size_t with
         Address => Size'Address,
         Import;
      Local_Tmp_3 : unsigned with
         Address => Flags'Address,
         Import;
      Local_Tmp_0 : driver_types_h.cudaError_t  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/cuda_runtime_api.h:4575
      := cuda_runtime_api_h.cudaHostAlloc (Local_Tmp_1, Local_Tmp_2, Local_Tmp_3);

   begin
      if Local_Tmp_0 /= 0 then
         Ada.Exceptions.Raise_Exception (CUDA.Exception_Registry.Element (Integer (Local_Tmp_0)));
      end if;
   end Host_Alloc;

   procedure Host_Register (Ptr : System.Address; Size : CUDA.Crtdefs.Size_T; Flags : unsigned) is

      Local_Tmp_1 : System.Address with
         Address => Ptr'Address,
         Import;
      Local_Tmp_2 : crtdefs_h.size_t with
         Address => Size'Address,
         Import;
      Local_Tmp_3 : unsigned with
         Address => Flags'Address,
         Import;
      Local_Tmp_0 : driver_types_h.cudaError_t  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/cuda_runtime_api.h:4659
      := cuda_runtime_api_h.cudaHostRegister (Local_Tmp_1, Local_Tmp_2, Local_Tmp_3);

   begin
      if Local_Tmp_0 /= 0 then
         Ada.Exceptions.Raise_Exception (CUDA.Exception_Registry.Element (Integer (Local_Tmp_0)));
      end if;
   end Host_Register;

   procedure Host_Unregister (Ptr : System.Address) is

      Local_Tmp_1 : System.Address with
         Address => Ptr'Address,
         Import;
      Local_Tmp_0 : driver_types_h.cudaError_t  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/cuda_runtime_api.h:4682
      := cuda_runtime_api_h.cudaHostUnregister (Local_Tmp_1);

   begin
      if Local_Tmp_0 /= 0 then
         Ada.Exceptions.Raise_Exception (CUDA.Exception_Registry.Element (Integer (Local_Tmp_0)));
      end if;
   end Host_Unregister;

   procedure Host_Get_Device_Pointer (P_Device : System.Address; P_Host : System.Address; Flags : unsigned) is

      Local_Tmp_1 : System.Address with
         Address => P_Device'Address,
         Import;
      Local_Tmp_2 : System.Address with
         Address => P_Host'Address,
         Import;
      Local_Tmp_3 : unsigned with
         Address => Flags'Address,
         Import;
      Local_Tmp_0 : driver_types_h.cudaError_t  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/cuda_runtime_api.h:4727
      := cuda_runtime_api_h.cudaHostGetDevicePointer (Local_Tmp_1, Local_Tmp_2, Local_Tmp_3);

   begin
      if Local_Tmp_0 /= 0 then
         Ada.Exceptions.Raise_Exception (CUDA.Exception_Registry.Element (Integer (Local_Tmp_0)));
      end if;
   end Host_Get_Device_Pointer;

   function Host_Get_Flags (P_Host : System.Address) return unsigned is

      Local_Tmp_1 : aliased unsigned;
      Local_Tmp_2 : aliased unsigned with
         Address => Local_Tmp_1'Address,
         Import;
      Local_Tmp_3 : System.Address with
         Address => P_Host'Address,
         Import;
      Local_Tmp_0 : driver_types_h.cudaError_t  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/cuda_runtime_api.h:4749
      := cuda_runtime_api_h.cudaHostGetFlags (Local_Tmp_1'Unchecked_Access, Local_Tmp_3);

   begin
      if Local_Tmp_0 /= 0 then
         Ada.Exceptions.Raise_Exception (CUDA.Exception_Registry.Element (Integer (Local_Tmp_0)));
      end if;
      return Local_Tmp_2;
   end Host_Get_Flags;

   procedure Malloc3_D (Pitched_Dev_Ptr : out CUDA.Driver_Types.Pitched_Ptr; Extent : CUDA.Driver_Types.Extent_T) is

      Local_Tmp_1 : aliased driver_types_h.cudaPitchedPtr with
         Address => Pitched_Dev_Ptr'Address,
         Import;
      Local_Tmp_2 : driver_types_h.cudaExtent with
         Address => Extent'Address,
         Import;
      Local_Tmp_0 : driver_types_h.cudaError_t  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/cuda_runtime_api.h:4788
      := cuda_runtime_api_h.cudaMalloc3D (Local_Tmp_1'Access, Local_Tmp_2);

   begin
      if Local_Tmp_0 /= 0 then
         Ada.Exceptions.Raise_Exception (CUDA.Exception_Registry.Element (Integer (Local_Tmp_0)));
      end if;
   end Malloc3_D;

   procedure Malloc3_DArray (C_Array : System.Address; Desc : out CUDA.Driver_Types.Channel_Format_Desc; Extent : CUDA.Driver_Types.Extent_T; Flags : unsigned) is

      Local_Tmp_1 : System.Address with
         Address => C_Array'Address,
         Import;
      Local_Tmp_2 : aliased driver_types_h.cudaChannelFormatDesc with
         Address => Desc'Address,
         Import;
      Local_Tmp_3 : driver_types_h.cudaExtent with
         Address => Extent'Address,
         Import;
      Local_Tmp_4 : unsigned with
         Address => Flags'Address,
         Import;
      Local_Tmp_0 : driver_types_h.cudaError_t  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/cuda_runtime_api.h:4927
      := cuda_runtime_api_h.cudaMalloc3DArray (Local_Tmp_1, Local_Tmp_2'Access, Local_Tmp_3, Local_Tmp_4);

   begin
      if Local_Tmp_0 /= 0 then
         Ada.Exceptions.Raise_Exception (CUDA.Exception_Registry.Element (Integer (Local_Tmp_0)));
      end if;
   end Malloc3_DArray;

   procedure Malloc_Mipmapped_Array (Mipmapped_Array : System.Address; Desc : out CUDA.Driver_Types.Channel_Format_Desc; Extent : CUDA.Driver_Types.Extent_T; Num_Levels : unsigned; Flags : unsigned) is

      Local_Tmp_1 : System.Address with
         Address => Mipmapped_Array'Address,
         Import;
      Local_Tmp_2 : aliased driver_types_h.cudaChannelFormatDesc with
         Address => Desc'Address,
         Import;
      Local_Tmp_3 : driver_types_h.cudaExtent with
         Address => Extent'Address,
         Import;
      Local_Tmp_4 : unsigned with
         Address => Num_Levels'Address,
         Import;
      Local_Tmp_5 : unsigned with
         Address => Flags'Address,
         Import;
      Local_Tmp_0 : driver_types_h.cudaError_t  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/cuda_runtime_api.h:5066
      := cuda_runtime_api_h.cudaMallocMipmappedArray (Local_Tmp_1, Local_Tmp_2'Access, Local_Tmp_3, Local_Tmp_4, Local_Tmp_5);

   begin
      if Local_Tmp_0 /= 0 then
         Ada.Exceptions.Raise_Exception (CUDA.Exception_Registry.Element (Integer (Local_Tmp_0)));
      end if;
   end Malloc_Mipmapped_Array;

   procedure Get_Mipmapped_Array_Level (Level_Array : System.Address; Mipmapped_Array : CUDA.Driver_Types.Mipmapped_Array_Const_T; Level : unsigned) is

      Local_Tmp_1 : System.Address with
         Address => Level_Array'Address,
         Import;
      Local_Tmp_2 : driver_types_h.cudaMipmappedArray_const_t with
         Address => Mipmapped_Array'Address,
         Import;
      Local_Tmp_3 : unsigned with
         Address => Level'Address,
         Import;
      Local_Tmp_0 : driver_types_h.cudaError_t  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/cuda_runtime_api.h:5095
      := cuda_runtime_api_h.cudaGetMipmappedArrayLevel (Local_Tmp_1, Local_Tmp_2, Local_Tmp_3);

   begin
      if Local_Tmp_0 /= 0 then
         Ada.Exceptions.Raise_Exception (CUDA.Exception_Registry.Element (Integer (Local_Tmp_0)));
      end if;
   end Get_Mipmapped_Array_Level;

   procedure Memcpy3_D (P : out CUDA.Driver_Types.Memcpy3_DParms) is

      Local_Tmp_1 : aliased driver_types_h.cudaMemcpy3DParms with
         Address => P'Address,
         Import;
      Local_Tmp_0 : driver_types_h.cudaError_t  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/cuda_runtime_api.h:5200
      := cuda_runtime_api_h.cudaMemcpy3D (Local_Tmp_1'Access);

   begin
      if Local_Tmp_0 /= 0 then
         Ada.Exceptions.Raise_Exception (CUDA.Exception_Registry.Element (Integer (Local_Tmp_0)));
      end if;
   end Memcpy3_D;

   procedure Memcpy3_DPeer (P : out CUDA.Driver_Types.Memcpy3_DPeer_Parms) is

      Local_Tmp_1 : aliased driver_types_h.cudaMemcpy3DPeerParms with
         Address => P'Address,
         Import;
      Local_Tmp_0 : driver_types_h.cudaError_t  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/cuda_runtime_api.h:5231
      := cuda_runtime_api_h.cudaMemcpy3DPeer (Local_Tmp_1'Access);

   begin
      if Local_Tmp_0 /= 0 then
         Ada.Exceptions.Raise_Exception (CUDA.Exception_Registry.Element (Integer (Local_Tmp_0)));
      end if;
   end Memcpy3_DPeer;

   procedure Memcpy3_DAsync (P : out CUDA.Driver_Types.Memcpy3_DParms; Stream : CUDA.Driver_Types.Stream_T) is

      Local_Tmp_1 : aliased driver_types_h.cudaMemcpy3DParms with
         Address => P'Address,
         Import;
      Local_Tmp_2 : driver_types_h.cudaStream_t with
         Address => Stream'Address,
         Import;
      Local_Tmp_0 : driver_types_h.cudaError_t  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/cuda_runtime_api.h:5349
      := cuda_runtime_api_h.cudaMemcpy3DAsync (Local_Tmp_1'Access, Local_Tmp_2);

   begin
      if Local_Tmp_0 /= 0 then
         Ada.Exceptions.Raise_Exception (CUDA.Exception_Registry.Element (Integer (Local_Tmp_0)));
      end if;
   end Memcpy3_DAsync;

   procedure Memcpy3_DPeer_Async (P : out CUDA.Driver_Types.Memcpy3_DPeer_Parms; Stream : CUDA.Driver_Types.Stream_T) is

      Local_Tmp_1 : aliased driver_types_h.cudaMemcpy3DPeerParms with
         Address => P'Address,
         Import;
      Local_Tmp_2 : driver_types_h.cudaStream_t with
         Address => Stream'Address,
         Import;
      Local_Tmp_0 : driver_types_h.cudaError_t  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/cuda_runtime_api.h:5375
      := cuda_runtime_api_h.cudaMemcpy3DPeerAsync (Local_Tmp_1'Access, Local_Tmp_2);

   begin
      if Local_Tmp_0 /= 0 then
         Ada.Exceptions.Raise_Exception (CUDA.Exception_Registry.Element (Integer (Local_Tmp_0)));
      end if;
   end Memcpy3_DPeer_Async;

   function Mem_Get_Info (Total : out CUDA.Crtdefs.Size_T) return CUDA.Crtdefs.Size_T is

      Local_Tmp_1 : aliased crtdefs_h.size_t;
      Local_Tmp_2 : aliased CUDA.Crtdefs.Size_T with
         Address => Local_Tmp_1'Address,
         Import;
      Local_Tmp_3 : aliased crtdefs_h.size_t with
         Address => Total'Address,
         Import;
      Local_Tmp_0 : driver_types_h.cudaError_t  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/cuda_runtime_api.h:5397
      := cuda_runtime_api_h.cudaMemGetInfo (Local_Tmp_1'Unchecked_Access, Local_Tmp_3'Access);

   begin
      if Local_Tmp_0 /= 0 then
         Ada.Exceptions.Raise_Exception (CUDA.Exception_Registry.Element (Integer (Local_Tmp_0)));
      end if;
      return Local_Tmp_2;
   end Mem_Get_Info;

   function CUDA_Array_Get_Info (Extent : out CUDA.Driver_Types.Extent_T; Flags : out unsigned; C_Array : CUDA.Driver_Types.CUDA_Array_T) return CUDA.Driver_Types.Channel_Format_Desc is

      Local_Tmp_1 : aliased driver_types_h.cudaChannelFormatDesc;
      Local_Tmp_2 : aliased CUDA.Driver_Types.Channel_Format_Desc with
         Address => Local_Tmp_1'Address,
         Import;
      Local_Tmp_3 : aliased driver_types_h.cudaExtent with
         Address => Extent'Address,
         Import;
      Local_Tmp_4 : aliased unsigned with
         Address => Flags'Address,
         Import;
      Local_Tmp_5 : driver_types_h.cudaArray_t with
         Address => C_Array'Address,
         Import;
      Local_Tmp_0 : driver_types_h.cudaError_t  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/cuda_runtime_api.h:5423
      := cuda_runtime_api_h.cudaArrayGetInfo (Local_Tmp_1'Unchecked_Access, Local_Tmp_3'Access, Local_Tmp_4'Access, Local_Tmp_5);

   begin
      if Local_Tmp_0 /= 0 then
         Ada.Exceptions.Raise_Exception (CUDA.Exception_Registry.Element (Integer (Local_Tmp_0)));
      end if;
      return Local_Tmp_2;
   end CUDA_Array_Get_Info;

   procedure Memcpy (Dst : System.Address; Src : System.Address; Count : CUDA.Crtdefs.Size_T; Kind : CUDA.Driver_Types.Memcpy_Kind) is

      Local_Tmp_1 : System.Address with
         Address => Dst'Address,
         Import;
      Local_Tmp_2 : System.Address with
         Address => Src'Address,
         Import;
      Local_Tmp_3 : crtdefs_h.size_t with
         Address => Count'Address,
         Import;
      Local_Tmp_4 : driver_types_h.cudaMemcpyKind with
         Address => Kind'Address,
         Import;
      Local_Tmp_0 : driver_types_h.cudaError_t  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/cuda_runtime_api.h:5466
      := cuda_runtime_api_h.cudaMemcpy (Local_Tmp_1, Local_Tmp_2, Local_Tmp_3, Local_Tmp_4);

   begin
      if Local_Tmp_0 /= 0 then
         Ada.Exceptions.Raise_Exception (CUDA.Exception_Registry.Element (Integer (Local_Tmp_0)));
      end if;
   end Memcpy;

   procedure Memcpy_Peer (Dst : System.Address; Dst_Device : int; Src : System.Address; Src_Device : int; Count : CUDA.Crtdefs.Size_T) is

      Local_Tmp_1 : System.Address with
         Address => Dst'Address,
         Import;
      Local_Tmp_2 : int with
         Address => Dst_Device'Address,
         Import;
      Local_Tmp_3 : System.Address with
         Address => Src'Address,
         Import;
      Local_Tmp_4 : int with
         Address => Src_Device'Address,
         Import;
      Local_Tmp_5 : crtdefs_h.size_t with
         Address => Count'Address,
         Import;
      Local_Tmp_0 : driver_types_h.cudaError_t  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/cuda_runtime_api.h:5501
      := cuda_runtime_api_h.cudaMemcpyPeer (Local_Tmp_1, Local_Tmp_2, Local_Tmp_3, Local_Tmp_4, Local_Tmp_5);

   begin
      if Local_Tmp_0 /= 0 then
         Ada.Exceptions.Raise_Exception (CUDA.Exception_Registry.Element (Integer (Local_Tmp_0)));
      end if;
   end Memcpy_Peer;

   procedure Memcpy2_D (Dst : System.Address; Dpitch : CUDA.Crtdefs.Size_T; Src : System.Address; Spitch : CUDA.Crtdefs.Size_T; Width : CUDA.Crtdefs.Size_T; Height : CUDA.Crtdefs.Size_T; Kind : CUDA.Driver_Types.Memcpy_Kind) is

      Local_Tmp_1 : System.Address with
         Address => Dst'Address,
         Import;
      Local_Tmp_2 : crtdefs_h.size_t with
         Address => Dpitch'Address,
         Import;
      Local_Tmp_3 : System.Address with
         Address => Src'Address,
         Import;
      Local_Tmp_4 : crtdefs_h.size_t with
         Address => Spitch'Address,
         Import;
      Local_Tmp_5 : crtdefs_h.size_t with
         Address => Width'Address,
         Import;
      Local_Tmp_6 : crtdefs_h.size_t with
         Address => Height'Address,
         Import;
      Local_Tmp_7 : driver_types_h.cudaMemcpyKind with
         Address => Kind'Address,
         Import;
      Local_Tmp_0 : driver_types_h.cudaError_t  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/cuda_runtime_api.h:5549
      := cuda_runtime_api_h.cudaMemcpy2D (Local_Tmp_1, Local_Tmp_2, Local_Tmp_3, Local_Tmp_4, Local_Tmp_5, Local_Tmp_6, Local_Tmp_7);

   begin
      if Local_Tmp_0 /= 0 then
         Ada.Exceptions.Raise_Exception (CUDA.Exception_Registry.Element (Integer (Local_Tmp_0)));
      end if;
   end Memcpy2_D;

   procedure Memcpy2_DTo_Array (Dst : CUDA.Driver_Types.CUDA_Array_T; W_Offset : CUDA.Crtdefs.Size_T; H_Offset : CUDA.Crtdefs.Size_T; Src : System.Address; Spitch : CUDA.Crtdefs.Size_T; Width : CUDA.Crtdefs.Size_T; Height : CUDA.Crtdefs.Size_T; Kind : CUDA.Driver_Types.Memcpy_Kind) is

      Local_Tmp_1 : driver_types_h.cudaArray_t with
         Address => Dst'Address,
         Import;
      Local_Tmp_2 : crtdefs_h.size_t with
         Address => W_Offset'Address,
         Import;
      Local_Tmp_3 : crtdefs_h.size_t with
         Address => H_Offset'Address,
         Import;
      Local_Tmp_4 : System.Address with
         Address => Src'Address,
         Import;
      Local_Tmp_5 : crtdefs_h.size_t with
         Address => Spitch'Address,
         Import;
      Local_Tmp_6 : crtdefs_h.size_t with
         Address => Width'Address,
         Import;
      Local_Tmp_7 : crtdefs_h.size_t with
         Address => Height'Address,
         Import;
      Local_Tmp_8 : driver_types_h.cudaMemcpyKind with
         Address => Kind'Address,
         Import;
      Local_Tmp_0 : driver_types_h.cudaError_t  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/cuda_runtime_api.h:5598
      := cuda_runtime_api_h.cudaMemcpy2DToArray (Local_Tmp_1, Local_Tmp_2, Local_Tmp_3, Local_Tmp_4, Local_Tmp_5, Local_Tmp_6, Local_Tmp_7, Local_Tmp_8);

   begin
      if Local_Tmp_0 /= 0 then
         Ada.Exceptions.Raise_Exception (CUDA.Exception_Registry.Element (Integer (Local_Tmp_0)));
      end if;
   end Memcpy2_DTo_Array;

   procedure Memcpy2_DFrom_Array (Dst : System.Address; Dpitch : CUDA.Crtdefs.Size_T; Src : CUDA.Driver_Types.CUDA_Array_Const_T; W_Offset : CUDA.Crtdefs.Size_T; H_Offset : CUDA.Crtdefs.Size_T; Width : CUDA.Crtdefs.Size_T; Height : CUDA.Crtdefs.Size_T; Kind : CUDA.Driver_Types.Memcpy_Kind) is

      Local_Tmp_1 : System.Address with
         Address => Dst'Address,
         Import;
      Local_Tmp_2 : crtdefs_h.size_t with
         Address => Dpitch'Address,
         Import;
      Local_Tmp_3 : driver_types_h.cudaArray_const_t with
         Address => Src'Address,
         Import;
      Local_Tmp_4 : crtdefs_h.size_t with
         Address => W_Offset'Address,
         Import;
      Local_Tmp_5 : crtdefs_h.size_t with
         Address => H_Offset'Address,
         Import;
      Local_Tmp_6 : crtdefs_h.size_t with
         Address => Width'Address,
         Import;
      Local_Tmp_7 : crtdefs_h.size_t with
         Address => Height'Address,
         Import;
      Local_Tmp_8 : driver_types_h.cudaMemcpyKind with
         Address => Kind'Address,
         Import;
      Local_Tmp_0 : driver_types_h.cudaError_t  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/cuda_runtime_api.h:5647
      := cuda_runtime_api_h.cudaMemcpy2DFromArray (Local_Tmp_1, Local_Tmp_2, Local_Tmp_3, Local_Tmp_4, Local_Tmp_5, Local_Tmp_6, Local_Tmp_7, Local_Tmp_8);

   begin
      if Local_Tmp_0 /= 0 then
         Ada.Exceptions.Raise_Exception (CUDA.Exception_Registry.Element (Integer (Local_Tmp_0)));
      end if;
   end Memcpy2_DFrom_Array;

   procedure Memcpy2_DArray_To_Array
     (Dst  : CUDA.Driver_Types.CUDA_Array_T; W_Offset_Dst : CUDA.Crtdefs.Size_T; H_Offset_Dst : CUDA.Crtdefs.Size_T; Src : CUDA.Driver_Types.CUDA_Array_Const_T; W_Offset_Src : CUDA.Crtdefs.Size_T; H_Offset_Src : CUDA.Crtdefs.Size_T; Width : CUDA.Crtdefs.Size_T; Height : CUDA.Crtdefs.Size_T;
      Kind : CUDA.Driver_Types.Memcpy_Kind)
   is

      Local_Tmp_1 : driver_types_h.cudaArray_t with
         Address => Dst'Address,
         Import;
      Local_Tmp_2 : crtdefs_h.size_t with
         Address => W_Offset_Dst'Address,
         Import;
      Local_Tmp_3 : crtdefs_h.size_t with
         Address => H_Offset_Dst'Address,
         Import;
      Local_Tmp_4 : driver_types_h.cudaArray_const_t with
         Address => Src'Address,
         Import;
      Local_Tmp_5 : crtdefs_h.size_t with
         Address => W_Offset_Src'Address,
         Import;
      Local_Tmp_6 : crtdefs_h.size_t with
         Address => H_Offset_Src'Address,
         Import;
      Local_Tmp_7 : crtdefs_h.size_t with
         Address => Width'Address,
         Import;
      Local_Tmp_8 : crtdefs_h.size_t with
         Address => Height'Address,
         Import;
      Local_Tmp_9 : driver_types_h.cudaMemcpyKind with
         Address => Kind'Address,
         Import;
      Local_Tmp_0 : driver_types_h.cudaError_t  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/cuda_runtime_api.h:5694
      := cuda_runtime_api_h.cudaMemcpy2DArrayToArray (Local_Tmp_1, Local_Tmp_2, Local_Tmp_3, Local_Tmp_4, Local_Tmp_5, Local_Tmp_6, Local_Tmp_7, Local_Tmp_8, Local_Tmp_9);

   begin
      if Local_Tmp_0 /= 0 then
         Ada.Exceptions.Raise_Exception (CUDA.Exception_Registry.Element (Integer (Local_Tmp_0)));
      end if;
   end Memcpy2_DArray_To_Array;

   procedure Memcpy_To_Symbol (Symbol : System.Address; Src : System.Address; Count : CUDA.Crtdefs.Size_T; Offset : CUDA.Crtdefs.Size_T; Kind : CUDA.Driver_Types.Memcpy_Kind) is

      Local_Tmp_1 : System.Address with
         Address => Symbol'Address,
         Import;
      Local_Tmp_2 : System.Address with
         Address => Src'Address,
         Import;
      Local_Tmp_3 : crtdefs_h.size_t with
         Address => Count'Address,
         Import;
      Local_Tmp_4 : crtdefs_h.size_t with
         Address => Offset'Address,
         Import;
      Local_Tmp_5 : driver_types_h.cudaMemcpyKind with
         Address => Kind'Address,
         Import;
      Local_Tmp_0 : driver_types_h.cudaError_t  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/cuda_runtime_api.h:5737
      := cuda_runtime_api_h.cudaMemcpyToSymbol (Local_Tmp_1, Local_Tmp_2, Local_Tmp_3, Local_Tmp_4, Local_Tmp_5);

   begin
      if Local_Tmp_0 /= 0 then
         Ada.Exceptions.Raise_Exception (CUDA.Exception_Registry.Element (Integer (Local_Tmp_0)));
      end if;
   end Memcpy_To_Symbol;

   procedure Memcpy_From_Symbol (Dst : System.Address; Symbol : System.Address; Count : CUDA.Crtdefs.Size_T; Offset : CUDA.Crtdefs.Size_T; Kind : CUDA.Driver_Types.Memcpy_Kind) is

      Local_Tmp_1 : System.Address with
         Address => Dst'Address,
         Import;
      Local_Tmp_2 : System.Address with
         Address => Symbol'Address,
         Import;
      Local_Tmp_3 : crtdefs_h.size_t with
         Address => Count'Address,
         Import;
      Local_Tmp_4 : crtdefs_h.size_t with
         Address => Offset'Address,
         Import;
      Local_Tmp_5 : driver_types_h.cudaMemcpyKind with
         Address => Kind'Address,
         Import;
      Local_Tmp_0 : driver_types_h.cudaError_t  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/cuda_runtime_api.h:5780
      := cuda_runtime_api_h.cudaMemcpyFromSymbol (Local_Tmp_1, Local_Tmp_2, Local_Tmp_3, Local_Tmp_4, Local_Tmp_5);

   begin
      if Local_Tmp_0 /= 0 then
         Ada.Exceptions.Raise_Exception (CUDA.Exception_Registry.Element (Integer (Local_Tmp_0)));
      end if;
   end Memcpy_From_Symbol;

   procedure Memcpy_Async (Dst : System.Address; Src : System.Address; Count : CUDA.Crtdefs.Size_T; Kind : CUDA.Driver_Types.Memcpy_Kind; Stream : CUDA.Driver_Types.Stream_T) is

      Local_Tmp_1 : System.Address with
         Address => Dst'Address,
         Import;
      Local_Tmp_2 : System.Address with
         Address => Src'Address,
         Import;
      Local_Tmp_3 : crtdefs_h.size_t with
         Address => Count'Address,
         Import;
      Local_Tmp_4 : driver_types_h.cudaMemcpyKind with
         Address => Kind'Address,
         Import;
      Local_Tmp_5 : driver_types_h.cudaStream_t with
         Address => Stream'Address,
         Import;
      Local_Tmp_0 : driver_types_h.cudaError_t  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/cuda_runtime_api.h:5836
      := cuda_runtime_api_h.cudaMemcpyAsync (Local_Tmp_1, Local_Tmp_2, Local_Tmp_3, Local_Tmp_4, Local_Tmp_5);

   begin
      if Local_Tmp_0 /= 0 then
         Ada.Exceptions.Raise_Exception (CUDA.Exception_Registry.Element (Integer (Local_Tmp_0)));
      end if;
   end Memcpy_Async;

   procedure Memcpy_Peer_Async (Dst : System.Address; Dst_Device : int; Src : System.Address; Src_Device : int; Count : CUDA.Crtdefs.Size_T; Stream : CUDA.Driver_Types.Stream_T) is

      Local_Tmp_1 : System.Address with
         Address => Dst'Address,
         Import;
      Local_Tmp_2 : int with
         Address => Dst_Device'Address,
         Import;
      Local_Tmp_3 : System.Address with
         Address => Src'Address,
         Import;
      Local_Tmp_4 : int with
         Address => Src_Device'Address,
         Import;
      Local_Tmp_5 : crtdefs_h.size_t with
         Address => Count'Address,
         Import;
      Local_Tmp_6 : driver_types_h.cudaStream_t with
         Address => Stream'Address,
         Import;
      Local_Tmp_0 : driver_types_h.cudaError_t  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/cuda_runtime_api.h:5871
      := cuda_runtime_api_h.cudaMemcpyPeerAsync (Local_Tmp_1, Local_Tmp_2, Local_Tmp_3, Local_Tmp_4, Local_Tmp_5, Local_Tmp_6);

   begin
      if Local_Tmp_0 /= 0 then
         Ada.Exceptions.Raise_Exception (CUDA.Exception_Registry.Element (Integer (Local_Tmp_0)));
      end if;
   end Memcpy_Peer_Async;

   procedure Memcpy2_DAsync (Dst : System.Address; Dpitch : CUDA.Crtdefs.Size_T; Src : System.Address; Spitch : CUDA.Crtdefs.Size_T; Width : CUDA.Crtdefs.Size_T; Height : CUDA.Crtdefs.Size_T; Kind : CUDA.Driver_Types.Memcpy_Kind; Stream : CUDA.Driver_Types.Stream_T) is

      Local_Tmp_1 : System.Address with
         Address => Dst'Address,
         Import;
      Local_Tmp_2 : crtdefs_h.size_t with
         Address => Dpitch'Address,
         Import;
      Local_Tmp_3 : System.Address with
         Address => Src'Address,
         Import;
      Local_Tmp_4 : crtdefs_h.size_t with
         Address => Spitch'Address,
         Import;
      Local_Tmp_5 : crtdefs_h.size_t with
         Address => Width'Address,
         Import;
      Local_Tmp_6 : crtdefs_h.size_t with
         Address => Height'Address,
         Import;
      Local_Tmp_7 : driver_types_h.cudaMemcpyKind with
         Address => Kind'Address,
         Import;
      Local_Tmp_8 : driver_types_h.cudaStream_t with
         Address => Stream'Address,
         Import;
      Local_Tmp_0 : driver_types_h.cudaError_t  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/cuda_runtime_api.h:5933
      := cuda_runtime_api_h.cudaMemcpy2DAsync (Local_Tmp_1, Local_Tmp_2, Local_Tmp_3, Local_Tmp_4, Local_Tmp_5, Local_Tmp_6, Local_Tmp_7, Local_Tmp_8);

   begin
      if Local_Tmp_0 /= 0 then
         Ada.Exceptions.Raise_Exception (CUDA.Exception_Registry.Element (Integer (Local_Tmp_0)));
      end if;
   end Memcpy2_DAsync;

   procedure Memcpy2_DTo_Array_Async
     (Dst : CUDA.Driver_Types.CUDA_Array_T; W_Offset : CUDA.Crtdefs.Size_T; H_Offset : CUDA.Crtdefs.Size_T; Src : System.Address; Spitch : CUDA.Crtdefs.Size_T; Width : CUDA.Crtdefs.Size_T; Height : CUDA.Crtdefs.Size_T; Kind : CUDA.Driver_Types.Memcpy_Kind; Stream : CUDA.Driver_Types.Stream_T)
   is

      Local_Tmp_1 : driver_types_h.cudaArray_t with
         Address => Dst'Address,
         Import;
      Local_Tmp_2 : crtdefs_h.size_t with
         Address => W_Offset'Address,
         Import;
      Local_Tmp_3 : crtdefs_h.size_t with
         Address => H_Offset'Address,
         Import;
      Local_Tmp_4 : System.Address with
         Address => Src'Address,
         Import;
      Local_Tmp_5 : crtdefs_h.size_t with
         Address => Spitch'Address,
         Import;
      Local_Tmp_6 : crtdefs_h.size_t with
         Address => Width'Address,
         Import;
      Local_Tmp_7 : crtdefs_h.size_t with
         Address => Height'Address,
         Import;
      Local_Tmp_8 : driver_types_h.cudaMemcpyKind with
         Address => Kind'Address,
         Import;
      Local_Tmp_9 : driver_types_h.cudaStream_t with
         Address => Stream'Address,
         Import;
      Local_Tmp_0 : driver_types_h.cudaError_t  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/cuda_runtime_api.h:5990
      := cuda_runtime_api_h.cudaMemcpy2DToArrayAsync (Local_Tmp_1, Local_Tmp_2, Local_Tmp_3, Local_Tmp_4, Local_Tmp_5, Local_Tmp_6, Local_Tmp_7, Local_Tmp_8, Local_Tmp_9);

   begin
      if Local_Tmp_0 /= 0 then
         Ada.Exceptions.Raise_Exception (CUDA.Exception_Registry.Element (Integer (Local_Tmp_0)));
      end if;
   end Memcpy2_DTo_Array_Async;

   procedure Memcpy2_DFrom_Array_Async
     (Dst : System.Address; Dpitch : CUDA.Crtdefs.Size_T; Src : CUDA.Driver_Types.CUDA_Array_Const_T; W_Offset : CUDA.Crtdefs.Size_T; H_Offset : CUDA.Crtdefs.Size_T; Width : CUDA.Crtdefs.Size_T; Height : CUDA.Crtdefs.Size_T; Kind : CUDA.Driver_Types.Memcpy_Kind; Stream : CUDA.Driver_Types.Stream_T)
   is

      Local_Tmp_1 : System.Address with
         Address => Dst'Address,
         Import;
      Local_Tmp_2 : crtdefs_h.size_t with
         Address => Dpitch'Address,
         Import;
      Local_Tmp_3 : driver_types_h.cudaArray_const_t with
         Address => Src'Address,
         Import;
      Local_Tmp_4 : crtdefs_h.size_t with
         Address => W_Offset'Address,
         Import;
      Local_Tmp_5 : crtdefs_h.size_t with
         Address => H_Offset'Address,
         Import;
      Local_Tmp_6 : crtdefs_h.size_t with
         Address => Width'Address,
         Import;
      Local_Tmp_7 : crtdefs_h.size_t with
         Address => Height'Address,
         Import;
      Local_Tmp_8 : driver_types_h.cudaMemcpyKind with
         Address => Kind'Address,
         Import;
      Local_Tmp_9 : driver_types_h.cudaStream_t with
         Address => Stream'Address,
         Import;
      Local_Tmp_0 : driver_types_h.cudaError_t  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/cuda_runtime_api.h:6046
      := cuda_runtime_api_h.cudaMemcpy2DFromArrayAsync (Local_Tmp_1, Local_Tmp_2, Local_Tmp_3, Local_Tmp_4, Local_Tmp_5, Local_Tmp_6, Local_Tmp_7, Local_Tmp_8, Local_Tmp_9);

   begin
      if Local_Tmp_0 /= 0 then
         Ada.Exceptions.Raise_Exception (CUDA.Exception_Registry.Element (Integer (Local_Tmp_0)));
      end if;
   end Memcpy2_DFrom_Array_Async;

   procedure Memcpy_To_Symbol_Async (Symbol : System.Address; Src : System.Address; Count : CUDA.Crtdefs.Size_T; Offset : CUDA.Crtdefs.Size_T; Kind : CUDA.Driver_Types.Memcpy_Kind; Stream : CUDA.Driver_Types.Stream_T) is

      Local_Tmp_1 : System.Address with
         Address => Symbol'Address,
         Import;
      Local_Tmp_2 : System.Address with
         Address => Src'Address,
         Import;
      Local_Tmp_3 : crtdefs_h.size_t with
         Address => Count'Address,
         Import;
      Local_Tmp_4 : crtdefs_h.size_t with
         Address => Offset'Address,
         Import;
      Local_Tmp_5 : driver_types_h.cudaMemcpyKind with
         Address => Kind'Address,
         Import;
      Local_Tmp_6 : driver_types_h.cudaStream_t with
         Address => Stream'Address,
         Import;
      Local_Tmp_0 : driver_types_h.cudaError_t  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/cuda_runtime_api.h:6097
      := cuda_runtime_api_h.cudaMemcpyToSymbolAsync (Local_Tmp_1, Local_Tmp_2, Local_Tmp_3, Local_Tmp_4, Local_Tmp_5, Local_Tmp_6);

   begin
      if Local_Tmp_0 /= 0 then
         Ada.Exceptions.Raise_Exception (CUDA.Exception_Registry.Element (Integer (Local_Tmp_0)));
      end if;
   end Memcpy_To_Symbol_Async;

   procedure Memcpy_From_Symbol_Async (Dst : System.Address; Symbol : System.Address; Count : CUDA.Crtdefs.Size_T; Offset : CUDA.Crtdefs.Size_T; Kind : CUDA.Driver_Types.Memcpy_Kind; Stream : CUDA.Driver_Types.Stream_T) is

      Local_Tmp_1 : System.Address with
         Address => Dst'Address,
         Import;
      Local_Tmp_2 : System.Address with
         Address => Symbol'Address,
         Import;
      Local_Tmp_3 : crtdefs_h.size_t with
         Address => Count'Address,
         Import;
      Local_Tmp_4 : crtdefs_h.size_t with
         Address => Offset'Address,
         Import;
      Local_Tmp_5 : driver_types_h.cudaMemcpyKind with
         Address => Kind'Address,
         Import;
      Local_Tmp_6 : driver_types_h.cudaStream_t with
         Address => Stream'Address,
         Import;
      Local_Tmp_0 : driver_types_h.cudaError_t  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/cuda_runtime_api.h:6148
      := cuda_runtime_api_h.cudaMemcpyFromSymbolAsync (Local_Tmp_1, Local_Tmp_2, Local_Tmp_3, Local_Tmp_4, Local_Tmp_5, Local_Tmp_6);

   begin
      if Local_Tmp_0 /= 0 then
         Ada.Exceptions.Raise_Exception (CUDA.Exception_Registry.Element (Integer (Local_Tmp_0)));
      end if;
   end Memcpy_From_Symbol_Async;

   procedure Memset (Dev_Ptr : System.Address; Value : int; Count : CUDA.Crtdefs.Size_T) is

      Local_Tmp_1 : System.Address with
         Address => Dev_Ptr'Address,
         Import;
      Local_Tmp_2 : int with
         Address => Value'Address,
         Import;
      Local_Tmp_3 : crtdefs_h.size_t with
         Address => Count'Address,
         Import;
      Local_Tmp_0 : driver_types_h.cudaError_t  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/cuda_runtime_api.h:6177
      := cuda_runtime_api_h.cudaMemset (Local_Tmp_1, Local_Tmp_2, Local_Tmp_3);

   begin
      if Local_Tmp_0 /= 0 then
         Ada.Exceptions.Raise_Exception (CUDA.Exception_Registry.Element (Integer (Local_Tmp_0)));
      end if;
   end Memset;

   procedure Memset2_D (Dev_Ptr : System.Address; Pitch : CUDA.Crtdefs.Size_T; Value : int; Width : CUDA.Crtdefs.Size_T; Height : CUDA.Crtdefs.Size_T) is

      Local_Tmp_1 : System.Address with
         Address => Dev_Ptr'Address,
         Import;
      Local_Tmp_2 : crtdefs_h.size_t with
         Address => Pitch'Address,
         Import;
      Local_Tmp_3 : int with
         Address => Value'Address,
         Import;
      Local_Tmp_4 : crtdefs_h.size_t with
         Address => Width'Address,
         Import;
      Local_Tmp_5 : crtdefs_h.size_t with
         Address => Height'Address,
         Import;
      Local_Tmp_0 : driver_types_h.cudaError_t  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/cuda_runtime_api.h:6211
      := cuda_runtime_api_h.cudaMemset2D (Local_Tmp_1, Local_Tmp_2, Local_Tmp_3, Local_Tmp_4, Local_Tmp_5);

   begin
      if Local_Tmp_0 /= 0 then
         Ada.Exceptions.Raise_Exception (CUDA.Exception_Registry.Element (Integer (Local_Tmp_0)));
      end if;
   end Memset2_D;

   procedure Memset3_D (Pitched_Dev_Ptr : CUDA.Driver_Types.Pitched_Ptr; Value : int; Extent : CUDA.Driver_Types.Extent_T) is

      Local_Tmp_1 : driver_types_h.cudaPitchedPtr with
         Address => Pitched_Dev_Ptr'Address,
         Import;
      Local_Tmp_2 : int with
         Address => Value'Address,
         Import;
      Local_Tmp_3 : driver_types_h.cudaExtent with
         Address => Extent'Address,
         Import;
      Local_Tmp_0 : driver_types_h.cudaError_t  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/cuda_runtime_api.h:6255
      := cuda_runtime_api_h.cudaMemset3D (Local_Tmp_1, Local_Tmp_2, Local_Tmp_3);

   begin
      if Local_Tmp_0 /= 0 then
         Ada.Exceptions.Raise_Exception (CUDA.Exception_Registry.Element (Integer (Local_Tmp_0)));
      end if;
   end Memset3_D;

   procedure Memset_Async (Dev_Ptr : System.Address; Value : int; Count : CUDA.Crtdefs.Size_T; Stream : CUDA.Driver_Types.Stream_T) is

      Local_Tmp_1 : System.Address with
         Address => Dev_Ptr'Address,
         Import;
      Local_Tmp_2 : int with
         Address => Value'Address,
         Import;
      Local_Tmp_3 : crtdefs_h.size_t with
         Address => Count'Address,
         Import;
      Local_Tmp_4 : driver_types_h.cudaStream_t with
         Address => Stream'Address,
         Import;
      Local_Tmp_0 : driver_types_h.cudaError_t  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/cuda_runtime_api.h:6291
      := cuda_runtime_api_h.cudaMemsetAsync (Local_Tmp_1, Local_Tmp_2, Local_Tmp_3, Local_Tmp_4);

   begin
      if Local_Tmp_0 /= 0 then
         Ada.Exceptions.Raise_Exception (CUDA.Exception_Registry.Element (Integer (Local_Tmp_0)));
      end if;
   end Memset_Async;

   procedure Memset2_DAsync (Dev_Ptr : System.Address; Pitch : CUDA.Crtdefs.Size_T; Value : int; Width : CUDA.Crtdefs.Size_T; Height : CUDA.Crtdefs.Size_T; Stream : CUDA.Driver_Types.Stream_T) is

      Local_Tmp_1 : System.Address with
         Address => Dev_Ptr'Address,
         Import;
      Local_Tmp_2 : crtdefs_h.size_t with
         Address => Pitch'Address,
         Import;
      Local_Tmp_3 : int with
         Address => Value'Address,
         Import;
      Local_Tmp_4 : crtdefs_h.size_t with
         Address => Width'Address,
         Import;
      Local_Tmp_5 : crtdefs_h.size_t with
         Address => Height'Address,
         Import;
      Local_Tmp_6 : driver_types_h.cudaStream_t with
         Address => Stream'Address,
         Import;
      Local_Tmp_0 : driver_types_h.cudaError_t  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/cuda_runtime_api.h:6332
      := cuda_runtime_api_h.cudaMemset2DAsync (Local_Tmp_1, Local_Tmp_2, Local_Tmp_3, Local_Tmp_4, Local_Tmp_5, Local_Tmp_6);

   begin
      if Local_Tmp_0 /= 0 then
         Ada.Exceptions.Raise_Exception (CUDA.Exception_Registry.Element (Integer (Local_Tmp_0)));
      end if;
   end Memset2_DAsync;

   procedure Memset3_DAsync (Pitched_Dev_Ptr : CUDA.Driver_Types.Pitched_Ptr; Value : int; Extent : CUDA.Driver_Types.Extent_T; Stream : CUDA.Driver_Types.Stream_T) is

      Local_Tmp_1 : driver_types_h.cudaPitchedPtr with
         Address => Pitched_Dev_Ptr'Address,
         Import;
      Local_Tmp_2 : int with
         Address => Value'Address,
         Import;
      Local_Tmp_3 : driver_types_h.cudaExtent with
         Address => Extent'Address,
         Import;
      Local_Tmp_4 : driver_types_h.cudaStream_t with
         Address => Stream'Address,
         Import;
      Local_Tmp_0 : driver_types_h.cudaError_t  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/cuda_runtime_api.h:6383
      := cuda_runtime_api_h.cudaMemset3DAsync (Local_Tmp_1, Local_Tmp_2, Local_Tmp_3, Local_Tmp_4);

   begin
      if Local_Tmp_0 /= 0 then
         Ada.Exceptions.Raise_Exception (CUDA.Exception_Registry.Element (Integer (Local_Tmp_0)));
      end if;
   end Memset3_DAsync;

   procedure Get_Symbol_Address (Dev_Ptr : System.Address; Symbol : System.Address) is

      Local_Tmp_1 : System.Address with
         Address => Dev_Ptr'Address,
         Import;
      Local_Tmp_2 : System.Address with
         Address => Symbol'Address,
         Import;
      Local_Tmp_0 : driver_types_h.cudaError_t  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/cuda_runtime_api.h:6411
      := cuda_runtime_api_h.cudaGetSymbolAddress (Local_Tmp_1, Local_Tmp_2);

   begin
      if Local_Tmp_0 /= 0 then
         Ada.Exceptions.Raise_Exception (CUDA.Exception_Registry.Element (Integer (Local_Tmp_0)));
      end if;
   end Get_Symbol_Address;

   function Get_Symbol_Size (Symbol : System.Address) return CUDA.Crtdefs.Size_T is

      Local_Tmp_1 : aliased crtdefs_h.size_t;
      Local_Tmp_2 : aliased CUDA.Crtdefs.Size_T with
         Address => Local_Tmp_1'Address,
         Import;
      Local_Tmp_3 : System.Address with
         Address => Symbol'Address,
         Import;
      Local_Tmp_0 : driver_types_h.cudaError_t  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/cuda_runtime_api.h:6438
      := cuda_runtime_api_h.cudaGetSymbolSize (Local_Tmp_1'Unchecked_Access, Local_Tmp_3);

   begin
      if Local_Tmp_0 /= 0 then
         Ada.Exceptions.Raise_Exception (CUDA.Exception_Registry.Element (Integer (Local_Tmp_0)));
      end if;
      return Local_Tmp_2;
   end Get_Symbol_Size;

   procedure Mem_Prefetch_Async (Dev_Ptr : System.Address; Count : CUDA.Crtdefs.Size_T; Dst_Device : int; Stream : CUDA.Driver_Types.Stream_T) is

      Local_Tmp_1 : System.Address with
         Address => Dev_Ptr'Address,
         Import;
      Local_Tmp_2 : crtdefs_h.size_t with
         Address => Count'Address,
         Import;
      Local_Tmp_3 : int with
         Address => Dst_Device'Address,
         Import;
      Local_Tmp_4 : driver_types_h.cudaStream_t with
         Address => Stream'Address,
         Import;
      Local_Tmp_0 : driver_types_h.cudaError_t  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/cuda_runtime_api.h:6508
      := cuda_runtime_api_h.cudaMemPrefetchAsync (Local_Tmp_1, Local_Tmp_2, Local_Tmp_3, Local_Tmp_4);

   begin
      if Local_Tmp_0 /= 0 then
         Ada.Exceptions.Raise_Exception (CUDA.Exception_Registry.Element (Integer (Local_Tmp_0)));
      end if;
   end Mem_Prefetch_Async;

   procedure Mem_Advise (Dev_Ptr : System.Address; Count : CUDA.Crtdefs.Size_T; Advice : CUDA.Driver_Types.Memory_Advise; Device : int) is

      Local_Tmp_1 : System.Address with
         Address => Dev_Ptr'Address,
         Import;
      Local_Tmp_2 : crtdefs_h.size_t with
         Address => Count'Address,
         Import;
      Local_Tmp_3 : driver_types_h.cudaMemoryAdvise with
         Address => Advice'Address,
         Import;
      Local_Tmp_4 : int with
         Address => Device'Address,
         Import;
      Local_Tmp_0 : driver_types_h.cudaError_t  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/cuda_runtime_api.h:6624
      := cuda_runtime_api_h.cudaMemAdvise (Local_Tmp_1, Local_Tmp_2, Local_Tmp_3, Local_Tmp_4);

   begin
      if Local_Tmp_0 /= 0 then
         Ada.Exceptions.Raise_Exception (CUDA.Exception_Registry.Element (Integer (Local_Tmp_0)));
      end if;
   end Mem_Advise;

   procedure Mem_Range_Get_Attribute (Data : System.Address; Data_Size : CUDA.Crtdefs.Size_T; Attribute : CUDA.Driver_Types.Mem_Range_Attribute; Dev_Ptr : System.Address; Count : CUDA.Crtdefs.Size_T) is

      Local_Tmp_1 : System.Address with
         Address => Data'Address,
         Import;
      Local_Tmp_2 : crtdefs_h.size_t with
         Address => Data_Size'Address,
         Import;
      Local_Tmp_3 : driver_types_h.cudaMemRangeAttribute with
         Address => Attribute'Address,
         Import;
      Local_Tmp_4 : System.Address with
         Address => Dev_Ptr'Address,
         Import;
      Local_Tmp_5 : crtdefs_h.size_t with
         Address => Count'Address,
         Import;
      Local_Tmp_0 : driver_types_h.cudaError_t  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/cuda_runtime_api.h:6683
      := cuda_runtime_api_h.cudaMemRangeGetAttribute (Local_Tmp_1, Local_Tmp_2, Local_Tmp_3, Local_Tmp_4, Local_Tmp_5);

   begin
      if Local_Tmp_0 /= 0 then
         Ada.Exceptions.Raise_Exception (CUDA.Exception_Registry.Element (Integer (Local_Tmp_0)));
      end if;
   end Mem_Range_Get_Attribute;

   procedure Mem_Range_Get_Attributes (Data : System.Address; Data_Sizes : out CUDA.Crtdefs.Size_T; Attributes : out CUDA.Driver_Types.Mem_Range_Attribute; Num_Attributes : CUDA.Crtdefs.Size_T; Dev_Ptr : System.Address; Count : CUDA.Crtdefs.Size_T) is

      Local_Tmp_1 : System.Address with
         Address => Data'Address,
         Import;
      Local_Tmp_2 : aliased crtdefs_h.size_t with
         Address => Data_Sizes'Address,
         Import;
      Local_Tmp_3 : aliased driver_types_h.cudaMemRangeAttribute with
         Address => Attributes'Address,
         Import;
      Local_Tmp_4 : crtdefs_h.size_t with
         Address => Num_Attributes'Address,
         Import;
      Local_Tmp_5 : System.Address with
         Address => Dev_Ptr'Address,
         Import;
      Local_Tmp_6 : crtdefs_h.size_t with
         Address => Count'Address,
         Import;
      Local_Tmp_0 : driver_types_h.cudaError_t  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/cuda_runtime_api.h:6722
      := cuda_runtime_api_h.cudaMemRangeGetAttributes (Local_Tmp_1, Local_Tmp_2'Access, Local_Tmp_3'Access, Local_Tmp_4, Local_Tmp_5, Local_Tmp_6);

   begin
      if Local_Tmp_0 /= 0 then
         Ada.Exceptions.Raise_Exception (CUDA.Exception_Registry.Element (Integer (Local_Tmp_0)));
      end if;
   end Mem_Range_Get_Attributes;

   procedure Memcpy_To_Array (Dst : CUDA.Driver_Types.CUDA_Array_T; W_Offset : CUDA.Crtdefs.Size_T; H_Offset : CUDA.Crtdefs.Size_T; Src : System.Address; Count : CUDA.Crtdefs.Size_T; Kind : CUDA.Driver_Types.Memcpy_Kind) is

      Local_Tmp_1 : driver_types_h.cudaArray_t with
         Address => Dst'Address,
         Import;
      Local_Tmp_2 : crtdefs_h.size_t with
         Address => W_Offset'Address,
         Import;
      Local_Tmp_3 : crtdefs_h.size_t with
         Address => H_Offset'Address,
         Import;
      Local_Tmp_4 : System.Address with
         Address => Src'Address,
         Import;
      Local_Tmp_5 : crtdefs_h.size_t with
         Address => Count'Address,
         Import;
      Local_Tmp_6 : driver_types_h.cudaMemcpyKind with
         Address => Kind'Address,
         Import;
      Local_Tmp_0 : driver_types_h.cudaError_t  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/cuda_runtime_api.h:6782
      := cuda_runtime_api_h.cudaMemcpyToArray (Local_Tmp_1, Local_Tmp_2, Local_Tmp_3, Local_Tmp_4, Local_Tmp_5, Local_Tmp_6);

   begin
      if Local_Tmp_0 /= 0 then
         Ada.Exceptions.Raise_Exception (CUDA.Exception_Registry.Element (Integer (Local_Tmp_0)));
      end if;
   end Memcpy_To_Array;

   procedure Memcpy_From_Array (Dst : System.Address; Src : CUDA.Driver_Types.CUDA_Array_Const_T; W_Offset : CUDA.Crtdefs.Size_T; H_Offset : CUDA.Crtdefs.Size_T; Count : CUDA.Crtdefs.Size_T; Kind : CUDA.Driver_Types.Memcpy_Kind) is

      Local_Tmp_1 : System.Address with
         Address => Dst'Address,
         Import;
      Local_Tmp_2 : driver_types_h.cudaArray_const_t with
         Address => Src'Address,
         Import;
      Local_Tmp_3 : crtdefs_h.size_t with
         Address => W_Offset'Address,
         Import;
      Local_Tmp_4 : crtdefs_h.size_t with
         Address => H_Offset'Address,
         Import;
      Local_Tmp_5 : crtdefs_h.size_t with
         Address => Count'Address,
         Import;
      Local_Tmp_6 : driver_types_h.cudaMemcpyKind with
         Address => Kind'Address,
         Import;
      Local_Tmp_0 : driver_types_h.cudaError_t  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/cuda_runtime_api.h:6824
      := cuda_runtime_api_h.cudaMemcpyFromArray (Local_Tmp_1, Local_Tmp_2, Local_Tmp_3, Local_Tmp_4, Local_Tmp_5, Local_Tmp_6);

   begin
      if Local_Tmp_0 /= 0 then
         Ada.Exceptions.Raise_Exception (CUDA.Exception_Registry.Element (Integer (Local_Tmp_0)));
      end if;
   end Memcpy_From_Array;

   procedure Memcpy_Array_To_Array
     (Dst : CUDA.Driver_Types.CUDA_Array_T; W_Offset_Dst : CUDA.Crtdefs.Size_T; H_Offset_Dst : CUDA.Crtdefs.Size_T; Src : CUDA.Driver_Types.CUDA_Array_Const_T; W_Offset_Src : CUDA.Crtdefs.Size_T; H_Offset_Src : CUDA.Crtdefs.Size_T; Count : CUDA.Crtdefs.Size_T; Kind : CUDA.Driver_Types.Memcpy_Kind)
   is

      Local_Tmp_1 : driver_types_h.cudaArray_t with
         Address => Dst'Address,
         Import;
      Local_Tmp_2 : crtdefs_h.size_t with
         Address => W_Offset_Dst'Address,
         Import;
      Local_Tmp_3 : crtdefs_h.size_t with
         Address => H_Offset_Dst'Address,
         Import;
      Local_Tmp_4 : driver_types_h.cudaArray_const_t with
         Address => Src'Address,
         Import;
      Local_Tmp_5 : crtdefs_h.size_t with
         Address => W_Offset_Src'Address,
         Import;
      Local_Tmp_6 : crtdefs_h.size_t with
         Address => H_Offset_Src'Address,
         Import;
      Local_Tmp_7 : crtdefs_h.size_t with
         Address => Count'Address,
         Import;
      Local_Tmp_8 : driver_types_h.cudaMemcpyKind with
         Address => Kind'Address,
         Import;
      Local_Tmp_0 : driver_types_h.cudaError_t  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/cuda_runtime_api.h:6867
      := cuda_runtime_api_h.cudaMemcpyArrayToArray (Local_Tmp_1, Local_Tmp_2, Local_Tmp_3, Local_Tmp_4, Local_Tmp_5, Local_Tmp_6, Local_Tmp_7, Local_Tmp_8);

   begin
      if Local_Tmp_0 /= 0 then
         Ada.Exceptions.Raise_Exception (CUDA.Exception_Registry.Element (Integer (Local_Tmp_0)));
      end if;
   end Memcpy_Array_To_Array;

   procedure Memcpy_To_Array_Async (Dst : CUDA.Driver_Types.CUDA_Array_T; W_Offset : CUDA.Crtdefs.Size_T; H_Offset : CUDA.Crtdefs.Size_T; Src : System.Address; Count : CUDA.Crtdefs.Size_T; Kind : CUDA.Driver_Types.Memcpy_Kind; Stream : CUDA.Driver_Types.Stream_T) is

      Local_Tmp_1 : driver_types_h.cudaArray_t with
         Address => Dst'Address,
         Import;
      Local_Tmp_2 : crtdefs_h.size_t with
         Address => W_Offset'Address,
         Import;
      Local_Tmp_3 : crtdefs_h.size_t with
         Address => H_Offset'Address,
         Import;
      Local_Tmp_4 : System.Address with
         Address => Src'Address,
         Import;
      Local_Tmp_5 : crtdefs_h.size_t with
         Address => Count'Address,
         Import;
      Local_Tmp_6 : driver_types_h.cudaMemcpyKind with
         Address => Kind'Address,
         Import;
      Local_Tmp_7 : driver_types_h.cudaStream_t with
         Address => Stream'Address,
         Import;
      Local_Tmp_0 : driver_types_h.cudaError_t  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/cuda_runtime_api.h:6918
      := cuda_runtime_api_h.cudaMemcpyToArrayAsync (Local_Tmp_1, Local_Tmp_2, Local_Tmp_3, Local_Tmp_4, Local_Tmp_5, Local_Tmp_6, Local_Tmp_7);

   begin
      if Local_Tmp_0 /= 0 then
         Ada.Exceptions.Raise_Exception (CUDA.Exception_Registry.Element (Integer (Local_Tmp_0)));
      end if;
   end Memcpy_To_Array_Async;

   procedure Memcpy_From_Array_Async (Dst : System.Address; Src : CUDA.Driver_Types.CUDA_Array_Const_T; W_Offset : CUDA.Crtdefs.Size_T; H_Offset : CUDA.Crtdefs.Size_T; Count : CUDA.Crtdefs.Size_T; Kind : CUDA.Driver_Types.Memcpy_Kind; Stream : CUDA.Driver_Types.Stream_T) is

      Local_Tmp_1 : System.Address with
         Address => Dst'Address,
         Import;
      Local_Tmp_2 : driver_types_h.cudaArray_const_t with
         Address => Src'Address,
         Import;
      Local_Tmp_3 : crtdefs_h.size_t with
         Address => W_Offset'Address,
         Import;
      Local_Tmp_4 : crtdefs_h.size_t with
         Address => H_Offset'Address,
         Import;
      Local_Tmp_5 : crtdefs_h.size_t with
         Address => Count'Address,
         Import;
      Local_Tmp_6 : driver_types_h.cudaMemcpyKind with
         Address => Kind'Address,
         Import;
      Local_Tmp_7 : driver_types_h.cudaStream_t with
         Address => Stream'Address,
         Import;
      Local_Tmp_0 : driver_types_h.cudaError_t  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/cuda_runtime_api.h:6968
      := cuda_runtime_api_h.cudaMemcpyFromArrayAsync (Local_Tmp_1, Local_Tmp_2, Local_Tmp_3, Local_Tmp_4, Local_Tmp_5, Local_Tmp_6, Local_Tmp_7);

   begin
      if Local_Tmp_0 /= 0 then
         Ada.Exceptions.Raise_Exception (CUDA.Exception_Registry.Element (Integer (Local_Tmp_0)));
      end if;
   end Memcpy_From_Array_Async;

   function Pointer_Get_Attributes (Ptr : System.Address) return CUDA.Driver_Types.Pointer_Attributes is

      Local_Tmp_1 : aliased driver_types_h.cudaPointerAttributes;
      Local_Tmp_2 : aliased CUDA.Driver_Types.Pointer_Attributes with
         Address => Local_Tmp_1'Address,
         Import;
      Local_Tmp_3 : System.Address with
         Address => Ptr'Address,
         Import;
      Local_Tmp_0 : driver_types_h.cudaError_t  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/cuda_runtime_api.h:7134
      := cuda_runtime_api_h.cudaPointerGetAttributes (Local_Tmp_1'Unchecked_Access, Local_Tmp_3);

   begin
      if Local_Tmp_0 /= 0 then
         Ada.Exceptions.Raise_Exception (CUDA.Exception_Registry.Element (Integer (Local_Tmp_0)));
      end if;
      return Local_Tmp_2;
   end Pointer_Get_Attributes;

   procedure Device_Can_Access_Peer (Can_Access_Peer : out int; Device : int; Peer_Device : int) is

      Local_Tmp_1 : aliased int with
         Address => Can_Access_Peer'Address,
         Import;
      Local_Tmp_2 : int with
         Address => Device'Address,
         Import;
      Local_Tmp_3 : int with
         Address => Peer_Device'Address,
         Import;
      Local_Tmp_0 : driver_types_h.cudaError_t  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/cuda_runtime_api.h:7175
      := cuda_runtime_api_h.cudaDeviceCanAccessPeer (Local_Tmp_1'Access, Local_Tmp_2, Local_Tmp_3);

   begin
      if Local_Tmp_0 /= 0 then
         Ada.Exceptions.Raise_Exception (CUDA.Exception_Registry.Element (Integer (Local_Tmp_0)));
      end if;
   end Device_Can_Access_Peer;

   procedure Device_Enable_Peer_Access (Peer_Device : int; Flags : unsigned) is

      Local_Tmp_1 : int with
         Address => Peer_Device'Address,
         Import;
      Local_Tmp_2 : unsigned with
         Address => Flags'Address,
         Import;
      Local_Tmp_0 : driver_types_h.cudaError_t  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/cuda_runtime_api.h:7217
      := cuda_runtime_api_h.cudaDeviceEnablePeerAccess (Local_Tmp_1, Local_Tmp_2);

   begin
      if Local_Tmp_0 /= 0 then
         Ada.Exceptions.Raise_Exception (CUDA.Exception_Registry.Element (Integer (Local_Tmp_0)));
      end if;
   end Device_Enable_Peer_Access;

   procedure Device_Disable_Peer_Access (Peer_Device : int) is

      Local_Tmp_1 : int with
         Address => Peer_Device'Address,
         Import;
      Local_Tmp_0 : driver_types_h.cudaError_t  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/cuda_runtime_api.h:7239
      := cuda_runtime_api_h.cudaDeviceDisablePeerAccess (Local_Tmp_1);

   begin
      if Local_Tmp_0 /= 0 then
         Ada.Exceptions.Raise_Exception (CUDA.Exception_Registry.Element (Integer (Local_Tmp_0)));
      end if;
   end Device_Disable_Peer_Access;

   procedure Graphics_Unregister_Resource (Resource : CUDA.Driver_Types.Graphics_Resource_T) is

      Local_Tmp_1 : driver_types_h.cudaGraphicsResource_t with
         Address => Resource'Address,
         Import;
      Local_Tmp_0 : driver_types_h.cudaError_t  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/cuda_runtime_api.h:7302
      := cuda_runtime_api_h.cudaGraphicsUnregisterResource (Local_Tmp_1);

   begin
      if Local_Tmp_0 /= 0 then
         Ada.Exceptions.Raise_Exception (CUDA.Exception_Registry.Element (Integer (Local_Tmp_0)));
      end if;
   end Graphics_Unregister_Resource;

   procedure Graphics_Resource_Set_Map_Flags (Resource : CUDA.Driver_Types.Graphics_Resource_T; Flags : unsigned) is

      Local_Tmp_1 : driver_types_h.cudaGraphicsResource_t with
         Address => Resource'Address,
         Import;
      Local_Tmp_2 : unsigned with
         Address => Flags'Address,
         Import;
      Local_Tmp_0 : driver_types_h.cudaError_t  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/cuda_runtime_api.h:7337
      := cuda_runtime_api_h.cudaGraphicsResourceSetMapFlags (Local_Tmp_1, Local_Tmp_2);

   begin
      if Local_Tmp_0 /= 0 then
         Ada.Exceptions.Raise_Exception (CUDA.Exception_Registry.Element (Integer (Local_Tmp_0)));
      end if;
   end Graphics_Resource_Set_Map_Flags;

   procedure Graphics_Map_Resources (Count : int; Resources : System.Address; Stream : CUDA.Driver_Types.Stream_T) is

      Local_Tmp_1 : int with
         Address => Count'Address,
         Import;
      Local_Tmp_2 : System.Address with
         Address => Resources'Address,
         Import;
      Local_Tmp_3 : driver_types_h.cudaStream_t with
         Address => Stream'Address,
         Import;
      Local_Tmp_0 : driver_types_h.cudaError_t  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/cuda_runtime_api.h:7376
      := cuda_runtime_api_h.cudaGraphicsMapResources (Local_Tmp_1, Local_Tmp_2, Local_Tmp_3);

   begin
      if Local_Tmp_0 /= 0 then
         Ada.Exceptions.Raise_Exception (CUDA.Exception_Registry.Element (Integer (Local_Tmp_0)));
      end if;
   end Graphics_Map_Resources;

   procedure Graphics_Unmap_Resources (Count : int; Resources : System.Address; Stream : CUDA.Driver_Types.Stream_T) is

      Local_Tmp_1 : int with
         Address => Count'Address,
         Import;
      Local_Tmp_2 : System.Address with
         Address => Resources'Address,
         Import;
      Local_Tmp_3 : driver_types_h.cudaStream_t with
         Address => Stream'Address,
         Import;
      Local_Tmp_0 : driver_types_h.cudaError_t  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/cuda_runtime_api.h:7411
      := cuda_runtime_api_h.cudaGraphicsUnmapResources (Local_Tmp_1, Local_Tmp_2, Local_Tmp_3);

   begin
      if Local_Tmp_0 /= 0 then
         Ada.Exceptions.Raise_Exception (CUDA.Exception_Registry.Element (Integer (Local_Tmp_0)));
      end if;
   end Graphics_Unmap_Resources;

   procedure Graphics_Resource_Get_Mapped_Pointer (Dev_Ptr : System.Address; Size : out CUDA.Crtdefs.Size_T; Resource : CUDA.Driver_Types.Graphics_Resource_T) is

      Local_Tmp_1 : System.Address with
         Address => Dev_Ptr'Address,
         Import;
      Local_Tmp_2 : aliased crtdefs_h.size_t with
         Address => Size'Address,
         Import;
      Local_Tmp_3 : driver_types_h.cudaGraphicsResource_t with
         Address => Resource'Address,
         Import;
      Local_Tmp_0 : driver_types_h.cudaError_t  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/cuda_runtime_api.h:7443
      := cuda_runtime_api_h.cudaGraphicsResourceGetMappedPointer (Local_Tmp_1, Local_Tmp_2'Access, Local_Tmp_3);

   begin
      if Local_Tmp_0 /= 0 then
         Ada.Exceptions.Raise_Exception (CUDA.Exception_Registry.Element (Integer (Local_Tmp_0)));
      end if;
   end Graphics_Resource_Get_Mapped_Pointer;

   procedure Graphics_Sub_Resource_Get_Mapped_Array (C_Array : System.Address; Resource : CUDA.Driver_Types.Graphics_Resource_T; Array_Index : unsigned; Mip_Level : unsigned) is

      Local_Tmp_1 : System.Address with
         Address => C_Array'Address,
         Import;
      Local_Tmp_2 : driver_types_h.cudaGraphicsResource_t with
         Address => Resource'Address,
         Import;
      Local_Tmp_3 : unsigned with
         Address => Array_Index'Address,
         Import;
      Local_Tmp_4 : unsigned with
         Address => Mip_Level'Address,
         Import;
      Local_Tmp_0 : driver_types_h.cudaError_t  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/cuda_runtime_api.h:7481
      := cuda_runtime_api_h.cudaGraphicsSubResourceGetMappedArray (Local_Tmp_1, Local_Tmp_2, Local_Tmp_3, Local_Tmp_4);

   begin
      if Local_Tmp_0 /= 0 then
         Ada.Exceptions.Raise_Exception (CUDA.Exception_Registry.Element (Integer (Local_Tmp_0)));
      end if;
   end Graphics_Sub_Resource_Get_Mapped_Array;

   procedure Graphics_Resource_Get_Mapped_Mipmapped_Array (Mipmapped_Array : System.Address; Resource : CUDA.Driver_Types.Graphics_Resource_T) is

      Local_Tmp_1 : System.Address with
         Address => Mipmapped_Array'Address,
         Import;
      Local_Tmp_2 : driver_types_h.cudaGraphicsResource_t with
         Address => Resource'Address,
         Import;
      Local_Tmp_0 : driver_types_h.cudaError_t  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/cuda_runtime_api.h:7510
      := cuda_runtime_api_h.cudaGraphicsResourceGetMappedMipmappedArray (Local_Tmp_1, Local_Tmp_2);

   begin
      if Local_Tmp_0 /= 0 then
         Ada.Exceptions.Raise_Exception (CUDA.Exception_Registry.Element (Integer (Local_Tmp_0)));
      end if;
   end Graphics_Resource_Get_Mapped_Mipmapped_Array;

   procedure Bind_Texture (Offset : out CUDA.Crtdefs.Size_T; Texref : out CUDA.Texture_Types.Texture_Reference; Dev_Ptr : System.Address; Desc : out CUDA.Driver_Types.Channel_Format_Desc; Size : CUDA.Crtdefs.Size_T) is

      Local_Tmp_1 : aliased crtdefs_h.size_t with
         Address => Offset'Address,
         Import;
      Local_Tmp_2 : aliased texture_types_h.textureReference with
         Address => Texref'Address,
         Import;
      Local_Tmp_3 : System.Address with
         Address => Dev_Ptr'Address,
         Import;
      Local_Tmp_4 : aliased driver_types_h.cudaChannelFormatDesc with
         Address => Desc'Address,
         Import;
      Local_Tmp_5 : crtdefs_h.size_t with
         Address => Size'Address,
         Import;
      Local_Tmp_0 : driver_types_h.cudaError_t  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/cuda_runtime_api.h:7581
      := cuda_runtime_api_h.cudaBindTexture (Local_Tmp_1'Access, Local_Tmp_2'Access, Local_Tmp_3, Local_Tmp_4'Access, Local_Tmp_5);

   begin
      if Local_Tmp_0 /= 0 then
         Ada.Exceptions.Raise_Exception (CUDA.Exception_Registry.Element (Integer (Local_Tmp_0)));
      end if;
   end Bind_Texture;

   procedure Bind_Texture2_D (Offset : out CUDA.Crtdefs.Size_T; Texref : out CUDA.Texture_Types.Texture_Reference; Dev_Ptr : System.Address; Desc : out CUDA.Driver_Types.Channel_Format_Desc; Width : CUDA.Crtdefs.Size_T; Height : CUDA.Crtdefs.Size_T; Pitch : CUDA.Crtdefs.Size_T) is

      Local_Tmp_1 : aliased crtdefs_h.size_t with
         Address => Offset'Address,
         Import;
      Local_Tmp_2 : aliased texture_types_h.textureReference with
         Address => Texref'Address,
         Import;
      Local_Tmp_3 : System.Address with
         Address => Dev_Ptr'Address,
         Import;
      Local_Tmp_4 : aliased driver_types_h.cudaChannelFormatDesc with
         Address => Desc'Address,
         Import;
      Local_Tmp_5 : crtdefs_h.size_t with
         Address => Width'Address,
         Import;
      Local_Tmp_6 : crtdefs_h.size_t with
         Address => Height'Address,
         Import;
      Local_Tmp_7 : crtdefs_h.size_t with
         Address => Pitch'Address,
         Import;
      Local_Tmp_0 : driver_types_h.cudaError_t  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/cuda_runtime_api.h:7640
      := cuda_runtime_api_h.cudaBindTexture2D (Local_Tmp_1'Access, Local_Tmp_2'Access, Local_Tmp_3, Local_Tmp_4'Access, Local_Tmp_5, Local_Tmp_6, Local_Tmp_7);

   begin
      if Local_Tmp_0 /= 0 then
         Ada.Exceptions.Raise_Exception (CUDA.Exception_Registry.Element (Integer (Local_Tmp_0)));
      end if;
   end Bind_Texture2_D;

   procedure Bind_Texture_To_Array (Texref : out CUDA.Texture_Types.Texture_Reference; C_Array : CUDA.Driver_Types.CUDA_Array_Const_T; Desc : out CUDA.Driver_Types.Channel_Format_Desc) is

      Local_Tmp_1 : aliased texture_types_h.textureReference with
         Address => Texref'Address,
         Import;
      Local_Tmp_2 : driver_types_h.cudaArray_const_t with
         Address => C_Array'Address,
         Import;
      Local_Tmp_3 : aliased driver_types_h.cudaChannelFormatDesc with
         Address => Desc'Address,
         Import;
      Local_Tmp_0 : driver_types_h.cudaError_t  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/cuda_runtime_api.h:7678
      := cuda_runtime_api_h.cudaBindTextureToArray (Local_Tmp_1'Access, Local_Tmp_2, Local_Tmp_3'Access);

   begin
      if Local_Tmp_0 /= 0 then
         Ada.Exceptions.Raise_Exception (CUDA.Exception_Registry.Element (Integer (Local_Tmp_0)));
      end if;
   end Bind_Texture_To_Array;

   procedure Bind_Texture_To_Mipmapped_Array (Texref : out CUDA.Texture_Types.Texture_Reference; Mipmapped_Array : CUDA.Driver_Types.Mipmapped_Array_Const_T; Desc : out CUDA.Driver_Types.Channel_Format_Desc) is

      Local_Tmp_1 : aliased texture_types_h.textureReference with
         Address => Texref'Address,
         Import;
      Local_Tmp_2 : driver_types_h.cudaMipmappedArray_const_t with
         Address => Mipmapped_Array'Address,
         Import;
      Local_Tmp_3 : aliased driver_types_h.cudaChannelFormatDesc with
         Address => Desc'Address,
         Import;
      Local_Tmp_0 : driver_types_h.cudaError_t  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/cuda_runtime_api.h:7718
      := cuda_runtime_api_h.cudaBindTextureToMipmappedArray (Local_Tmp_1'Access, Local_Tmp_2, Local_Tmp_3'Access);

   begin
      if Local_Tmp_0 /= 0 then
         Ada.Exceptions.Raise_Exception (CUDA.Exception_Registry.Element (Integer (Local_Tmp_0)));
      end if;
   end Bind_Texture_To_Mipmapped_Array;

   procedure Unbind_Texture (Texref : out CUDA.Texture_Types.Texture_Reference) is

      Local_Tmp_1 : aliased texture_types_h.textureReference with
         Address => Texref'Address,
         Import;
      Local_Tmp_0 : driver_types_h.cudaError_t  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/cuda_runtime_api.h:7744
      := cuda_runtime_api_h.cudaUnbindTexture (Local_Tmp_1'Access);

   begin
      if Local_Tmp_0 /= 0 then
         Ada.Exceptions.Raise_Exception (CUDA.Exception_Registry.Element (Integer (Local_Tmp_0)));
      end if;
   end Unbind_Texture;

   function Get_Texture_Alignment_Offset (Texref : out CUDA.Texture_Types.Texture_Reference) return CUDA.Crtdefs.Size_T is

      Local_Tmp_1 : aliased crtdefs_h.size_t;
      Local_Tmp_2 : aliased CUDA.Crtdefs.Size_T with
         Address => Local_Tmp_1'Address,
         Import;
      Local_Tmp_3 : aliased texture_types_h.textureReference with
         Address => Texref'Address,
         Import;
      Local_Tmp_0 : driver_types_h.cudaError_t  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/cuda_runtime_api.h:7773
      := cuda_runtime_api_h.cudaGetTextureAlignmentOffset (Local_Tmp_1'Unchecked_Access, Local_Tmp_3'Access);

   begin
      if Local_Tmp_0 /= 0 then
         Ada.Exceptions.Raise_Exception (CUDA.Exception_Registry.Element (Integer (Local_Tmp_0)));
      end if;
      return Local_Tmp_2;
   end Get_Texture_Alignment_Offset;

   procedure Get_Texture_Reference (Texref : System.Address; Symbol : System.Address) is

      Local_Tmp_1 : System.Address with
         Address => Texref'Address,
         Import;
      Local_Tmp_2 : System.Address with
         Address => Symbol'Address,
         Import;
      Local_Tmp_0 : driver_types_h.cudaError_t  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/cuda_runtime_api.h:7803
      := cuda_runtime_api_h.cudaGetTextureReference (Local_Tmp_1, Local_Tmp_2);

   begin
      if Local_Tmp_0 /= 0 then
         Ada.Exceptions.Raise_Exception (CUDA.Exception_Registry.Element (Integer (Local_Tmp_0)));
      end if;
   end Get_Texture_Reference;

   procedure Bind_Surface_To_Array (Surfref : out CUDA.Surface_Types.Surface_Reference; C_Array : CUDA.Driver_Types.CUDA_Array_Const_T; Desc : out CUDA.Driver_Types.Channel_Format_Desc) is

      Local_Tmp_1 : aliased surface_types_h.surfaceReference with
         Address => Surfref'Address,
         Import;
      Local_Tmp_2 : driver_types_h.cudaArray_const_t with
         Address => C_Array'Address,
         Import;
      Local_Tmp_3 : aliased driver_types_h.cudaChannelFormatDesc with
         Address => Desc'Address,
         Import;
      Local_Tmp_0 : driver_types_h.cudaError_t  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/cuda_runtime_api.h:7848
      := cuda_runtime_api_h.cudaBindSurfaceToArray (Local_Tmp_1'Access, Local_Tmp_2, Local_Tmp_3'Access);

   begin
      if Local_Tmp_0 /= 0 then
         Ada.Exceptions.Raise_Exception (CUDA.Exception_Registry.Element (Integer (Local_Tmp_0)));
      end if;
   end Bind_Surface_To_Array;

   procedure Get_Surface_Reference (Surfref : System.Address; Symbol : System.Address) is

      Local_Tmp_1 : System.Address with
         Address => Surfref'Address,
         Import;
      Local_Tmp_2 : System.Address with
         Address => Symbol'Address,
         Import;
      Local_Tmp_0 : driver_types_h.cudaError_t  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/cuda_runtime_api.h:7873
      := cuda_runtime_api_h.cudaGetSurfaceReference (Local_Tmp_1, Local_Tmp_2);

   begin
      if Local_Tmp_0 /= 0 then
         Ada.Exceptions.Raise_Exception (CUDA.Exception_Registry.Element (Integer (Local_Tmp_0)));
      end if;
   end Get_Surface_Reference;

   function Get_Channel_Desc (C_Array : CUDA.Driver_Types.CUDA_Array_Const_T) return CUDA.Driver_Types.Channel_Format_Desc is

      Local_Tmp_1 : aliased driver_types_h.cudaChannelFormatDesc;
      Local_Tmp_2 : aliased CUDA.Driver_Types.Channel_Format_Desc with
         Address => Local_Tmp_1'Address,
         Import;
      Local_Tmp_3 : driver_types_h.cudaArray_const_t with
         Address => C_Array'Address,
         Import;
      Local_Tmp_0 : driver_types_h.cudaError_t  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/cuda_runtime_api.h:7908
      := cuda_runtime_api_h.cudaGetChannelDesc (Local_Tmp_1'Unchecked_Access, Local_Tmp_3);

   begin
      if Local_Tmp_0 /= 0 then
         Ada.Exceptions.Raise_Exception (CUDA.Exception_Registry.Element (Integer (Local_Tmp_0)));
      end if;
      return Local_Tmp_2;
   end Get_Channel_Desc;

   function Create_Channel_Desc (X : int; Y : int; Z : int; W : int; F : CUDA.Driver_Types.Channel_Format_Kind) return CUDA.Driver_Types.Channel_Format_Desc is

      Local_Tmp_2 : int with
         Address => X'Address,
         Import;
      Local_Tmp_3 : int with
         Address => Y'Address,
         Import;
      Local_Tmp_4 : int with
         Address => Z'Address,
         Import;
      Local_Tmp_5 : int with
         Address => W'Address,
         Import;
      Local_Tmp_6 : driver_types_h.cudaChannelFormatKind with
         Address => F'Address,
         Import;
      Local_Tmp_1 : driver_types_h.cudaChannelFormatDesc  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/cuda_runtime_api.h:7938
      := cuda_runtime_api_h.cudaCreateChannelDesc (Local_Tmp_2, Local_Tmp_3, Local_Tmp_4, Local_Tmp_5, Local_Tmp_6);
      Local_Tmp_0 : CUDA.Driver_Types.Channel_Format_Desc with
         Address => Local_Tmp_1'Address,
         Import;

   begin
      return Local_Tmp_0;
   end Create_Channel_Desc;

   procedure Create_Texture_Object (P_Tex_Object : out CUDA.Texture_Types.Texture_Object_T; P_Res_Desc : out CUDA.Driver_Types.Resource_Desc; P_Tex_Desc : out CUDA.Texture_Types.Texture_Desc; P_Res_View_Desc : out CUDA.Driver_Types.Resource_View_Desc) is

      Local_Tmp_1 : aliased texture_types_h.cudaTextureObject_t with
         Address => P_Tex_Object'Address,
         Import;
      Local_Tmp_2 : aliased driver_types_h.cudaResourceDesc with
         Address => P_Res_Desc'Address,
         Import;
      Local_Tmp_3 : aliased texture_types_h.cudaTextureDesc with
         Address => P_Tex_Desc'Address,
         Import;
      Local_Tmp_4 : aliased driver_types_h.cudaResourceViewDesc with
         Address => P_Res_View_Desc'Address,
         Import;
      Local_Tmp_0 : driver_types_h.cudaError_t  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/cuda_runtime_api.h:8153
      := cuda_runtime_api_h.cudaCreateTextureObject (Local_Tmp_1'Access, Local_Tmp_2'Access, Local_Tmp_3'Access, Local_Tmp_4'Access);

   begin
      if Local_Tmp_0 /= 0 then
         Ada.Exceptions.Raise_Exception (CUDA.Exception_Registry.Element (Integer (Local_Tmp_0)));
      end if;
   end Create_Texture_Object;

   procedure Destroy_Texture_Object (Tex_Object : CUDA.Texture_Types.Texture_Object_T) is

      Local_Tmp_1 : texture_types_h.cudaTextureObject_t with
         Address => Tex_Object'Address,
         Import;
      Local_Tmp_0 : driver_types_h.cudaError_t  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/cuda_runtime_api.h:8172
      := cuda_runtime_api_h.cudaDestroyTextureObject (Local_Tmp_1);

   begin
      if Local_Tmp_0 /= 0 then
         Ada.Exceptions.Raise_Exception (CUDA.Exception_Registry.Element (Integer (Local_Tmp_0)));
      end if;
   end Destroy_Texture_Object;

   function Get_Texture_Object_Resource_Desc (Tex_Object : CUDA.Texture_Types.Texture_Object_T) return CUDA.Driver_Types.Resource_Desc is

      Local_Tmp_1 : aliased driver_types_h.cudaResourceDesc;
      Local_Tmp_2 : aliased CUDA.Driver_Types.Resource_Desc with
         Address => Local_Tmp_1'Address,
         Import;
      Local_Tmp_3 : texture_types_h.cudaTextureObject_t with
         Address => Tex_Object'Address,
         Import;
      Local_Tmp_0 : driver_types_h.cudaError_t  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/cuda_runtime_api.h:8192
      := cuda_runtime_api_h.cudaGetTextureObjectResourceDesc (Local_Tmp_1'Unchecked_Access, Local_Tmp_3);

   begin
      if Local_Tmp_0 /= 0 then
         Ada.Exceptions.Raise_Exception (CUDA.Exception_Registry.Element (Integer (Local_Tmp_0)));
      end if;
      return Local_Tmp_2;
   end Get_Texture_Object_Resource_Desc;

   function Get_Texture_Object_Texture_Desc (Tex_Object : CUDA.Texture_Types.Texture_Object_T) return CUDA.Texture_Types.Texture_Desc is

      Local_Tmp_1 : aliased texture_types_h.cudaTextureDesc;
      Local_Tmp_2 : aliased CUDA.Texture_Types.Texture_Desc with
         Address => Local_Tmp_1'Address,
         Import;
      Local_Tmp_3 : texture_types_h.cudaTextureObject_t with
         Address => Tex_Object'Address,
         Import;
      Local_Tmp_0 : driver_types_h.cudaError_t  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/cuda_runtime_api.h:8212
      := cuda_runtime_api_h.cudaGetTextureObjectTextureDesc (Local_Tmp_1'Unchecked_Access, Local_Tmp_3);

   begin
      if Local_Tmp_0 /= 0 then
         Ada.Exceptions.Raise_Exception (CUDA.Exception_Registry.Element (Integer (Local_Tmp_0)));
      end if;
      return Local_Tmp_2;
   end Get_Texture_Object_Texture_Desc;

   function Get_Texture_Object_Resource_View_Desc (Tex_Object : CUDA.Texture_Types.Texture_Object_T) return CUDA.Driver_Types.Resource_View_Desc is

      Local_Tmp_1 : aliased driver_types_h.cudaResourceViewDesc;
      Local_Tmp_2 : aliased CUDA.Driver_Types.Resource_View_Desc with
         Address => Local_Tmp_1'Address,
         Import;
      Local_Tmp_3 : texture_types_h.cudaTextureObject_t with
         Address => Tex_Object'Address,
         Import;
      Local_Tmp_0 : driver_types_h.cudaError_t  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/cuda_runtime_api.h:8233
      := cuda_runtime_api_h.cudaGetTextureObjectResourceViewDesc (Local_Tmp_1'Unchecked_Access, Local_Tmp_3);

   begin
      if Local_Tmp_0 /= 0 then
         Ada.Exceptions.Raise_Exception (CUDA.Exception_Registry.Element (Integer (Local_Tmp_0)));
      end if;
      return Local_Tmp_2;
   end Get_Texture_Object_Resource_View_Desc;

   procedure Create_Surface_Object (P_Surf_Object : out CUDA.Surface_Types.Surface_Object_T; P_Res_Desc : out CUDA.Driver_Types.Resource_Desc) is

      Local_Tmp_1 : aliased surface_types_h.cudaSurfaceObject_t with
         Address => P_Surf_Object'Address,
         Import;
      Local_Tmp_2 : aliased driver_types_h.cudaResourceDesc with
         Address => P_Res_Desc'Address,
         Import;
      Local_Tmp_0 : driver_types_h.cudaError_t  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/cuda_runtime_api.h:8278
      := cuda_runtime_api_h.cudaCreateSurfaceObject (Local_Tmp_1'Access, Local_Tmp_2'Access);

   begin
      if Local_Tmp_0 /= 0 then
         Ada.Exceptions.Raise_Exception (CUDA.Exception_Registry.Element (Integer (Local_Tmp_0)));
      end if;
   end Create_Surface_Object;

   procedure Destroy_Surface_Object (Surf_Object : CUDA.Surface_Types.Surface_Object_T) is

      Local_Tmp_1 : surface_types_h.cudaSurfaceObject_t with
         Address => Surf_Object'Address,
         Import;
      Local_Tmp_0 : driver_types_h.cudaError_t  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/cuda_runtime_api.h:8297
      := cuda_runtime_api_h.cudaDestroySurfaceObject (Local_Tmp_1);

   begin
      if Local_Tmp_0 /= 0 then
         Ada.Exceptions.Raise_Exception (CUDA.Exception_Registry.Element (Integer (Local_Tmp_0)));
      end if;
   end Destroy_Surface_Object;

   function Get_Surface_Object_Resource_Desc (Surf_Object : CUDA.Surface_Types.Surface_Object_T) return CUDA.Driver_Types.Resource_Desc is

      Local_Tmp_1 : aliased driver_types_h.cudaResourceDesc;
      Local_Tmp_2 : aliased CUDA.Driver_Types.Resource_Desc with
         Address => Local_Tmp_1'Address,
         Import;
      Local_Tmp_3 : surface_types_h.cudaSurfaceObject_t with
         Address => Surf_Object'Address,
         Import;
      Local_Tmp_0 : driver_types_h.cudaError_t  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/cuda_runtime_api.h:8316
      := cuda_runtime_api_h.cudaGetSurfaceObjectResourceDesc (Local_Tmp_1'Unchecked_Access, Local_Tmp_3);

   begin
      if Local_Tmp_0 /= 0 then
         Ada.Exceptions.Raise_Exception (CUDA.Exception_Registry.Element (Integer (Local_Tmp_0)));
      end if;
      return Local_Tmp_2;
   end Get_Surface_Object_Resource_Desc;

   function Driver_Get_Version return int is

      Local_Tmp_1 : aliased int;
      Local_Tmp_2 : aliased int with
         Address => Local_Tmp_1'Address,
         Import;
      Local_Tmp_0 : driver_types_h.cudaError_t  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/cuda_runtime_api.h:8350
      := cuda_runtime_api_h.cudaDriverGetVersion (Local_Tmp_1'Unchecked_Access);

   begin
      if Local_Tmp_0 /= 0 then
         Ada.Exceptions.Raise_Exception (CUDA.Exception_Registry.Element (Integer (Local_Tmp_0)));
      end if;
      return Local_Tmp_2;
   end Driver_Get_Version;

   function Runtime_Get_Version return int is

      Local_Tmp_1 : aliased int;
      Local_Tmp_2 : aliased int with
         Address => Local_Tmp_1'Address,
         Import;
      Local_Tmp_0 : driver_types_h.cudaError_t  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/cuda_runtime_api.h:8375
      := cuda_runtime_api_h.cudaRuntimeGetVersion (Local_Tmp_1'Unchecked_Access);

   begin
      if Local_Tmp_0 /= 0 then
         Ada.Exceptions.Raise_Exception (CUDA.Exception_Registry.Element (Integer (Local_Tmp_0)));
      end if;
      return Local_Tmp_2;
   end Runtime_Get_Version;

   procedure Graph_Create (P_Graph : System.Address; Flags : unsigned) is

      Local_Tmp_1 : System.Address with
         Address => P_Graph'Address,
         Import;
      Local_Tmp_2 : unsigned with
         Address => Flags'Address,
         Import;
      Local_Tmp_0 : driver_types_h.cudaError_t  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/cuda_runtime_api.h:8422
      := cuda_runtime_api_h.cudaGraphCreate (Local_Tmp_1, Local_Tmp_2);

   begin
      if Local_Tmp_0 /= 0 then
         Ada.Exceptions.Raise_Exception (CUDA.Exception_Registry.Element (Integer (Local_Tmp_0)));
      end if;
   end Graph_Create;

   procedure Graph_Add_Kernel_Node (P_Graph_Node : System.Address; Graph : CUDA.Driver_Types.Graph_T; P_Dependencies : System.Address; Num_Dependencies : CUDA.Crtdefs.Size_T; P_Node_Params : out CUDA.Driver_Types.Kernel_Node_Params) is

      Local_Tmp_1 : System.Address with
         Address => P_Graph_Node'Address,
         Import;
      Local_Tmp_2 : driver_types_h.cudaGraph_t with
         Address => Graph'Address,
         Import;
      Local_Tmp_3 : System.Address with
         Address => P_Dependencies'Address,
         Import;
      Local_Tmp_4 : crtdefs_h.size_t with
         Address => Num_Dependencies'Address,
         Import;
      Local_Tmp_5 : aliased driver_types_h.cudaKernelNodeParams with
         Address => P_Node_Params'Address,
         Import;
      Local_Tmp_0 : driver_types_h.cudaError_t  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/cuda_runtime_api.h:8519
      := cuda_runtime_api_h.cudaGraphAddKernelNode (Local_Tmp_1, Local_Tmp_2, Local_Tmp_3, Local_Tmp_4, Local_Tmp_5'Access);

   begin
      if Local_Tmp_0 /= 0 then
         Ada.Exceptions.Raise_Exception (CUDA.Exception_Registry.Element (Integer (Local_Tmp_0)));
      end if;
   end Graph_Add_Kernel_Node;

   procedure Graph_Kernel_Node_Get_Params (Node : CUDA.Driver_Types.Graph_Node_T; P_Node_Params : out CUDA.Driver_Types.Kernel_Node_Params) is

      Local_Tmp_1 : driver_types_h.cudaGraphNode_t with
         Address => Node'Address,
         Import;
      Local_Tmp_2 : aliased driver_types_h.cudaKernelNodeParams with
         Address => P_Node_Params'Address,
         Import;
      Local_Tmp_0 : driver_types_h.cudaError_t  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/cuda_runtime_api.h:8552
      := cuda_runtime_api_h.cudaGraphKernelNodeGetParams (Local_Tmp_1, Local_Tmp_2'Access);

   begin
      if Local_Tmp_0 /= 0 then
         Ada.Exceptions.Raise_Exception (CUDA.Exception_Registry.Element (Integer (Local_Tmp_0)));
      end if;
   end Graph_Kernel_Node_Get_Params;

   procedure Graph_Kernel_Node_Set_Params (Node : CUDA.Driver_Types.Graph_Node_T; P_Node_Params : out CUDA.Driver_Types.Kernel_Node_Params) is

      Local_Tmp_1 : driver_types_h.cudaGraphNode_t with
         Address => Node'Address,
         Import;
      Local_Tmp_2 : aliased driver_types_h.cudaKernelNodeParams with
         Address => P_Node_Params'Address,
         Import;
      Local_Tmp_0 : driver_types_h.cudaError_t  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/cuda_runtime_api.h:8577
      := cuda_runtime_api_h.cudaGraphKernelNodeSetParams (Local_Tmp_1, Local_Tmp_2'Access);

   begin
      if Local_Tmp_0 /= 0 then
         Ada.Exceptions.Raise_Exception (CUDA.Exception_Registry.Element (Integer (Local_Tmp_0)));
      end if;
   end Graph_Kernel_Node_Set_Params;

   procedure Graph_Add_Memcpy_Node (P_Graph_Node : System.Address; Graph : CUDA.Driver_Types.Graph_T; P_Dependencies : System.Address; Num_Dependencies : CUDA.Crtdefs.Size_T; P_Copy_Params : out CUDA.Driver_Types.Memcpy3_DParms) is

      Local_Tmp_1 : System.Address with
         Address => P_Graph_Node'Address,
         Import;
      Local_Tmp_2 : driver_types_h.cudaGraph_t with
         Address => Graph'Address,
         Import;
      Local_Tmp_3 : System.Address with
         Address => P_Dependencies'Address,
         Import;
      Local_Tmp_4 : crtdefs_h.size_t with
         Address => Num_Dependencies'Address,
         Import;
      Local_Tmp_5 : aliased driver_types_h.cudaMemcpy3DParms with
         Address => P_Copy_Params'Address,
         Import;
      Local_Tmp_0 : driver_types_h.cudaError_t  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/cuda_runtime_api.h:8621
      := cuda_runtime_api_h.cudaGraphAddMemcpyNode (Local_Tmp_1, Local_Tmp_2, Local_Tmp_3, Local_Tmp_4, Local_Tmp_5'Access);

   begin
      if Local_Tmp_0 /= 0 then
         Ada.Exceptions.Raise_Exception (CUDA.Exception_Registry.Element (Integer (Local_Tmp_0)));
      end if;
   end Graph_Add_Memcpy_Node;

   procedure Graph_Memcpy_Node_Get_Params (Node : CUDA.Driver_Types.Graph_Node_T; P_Node_Params : out CUDA.Driver_Types.Memcpy3_DParms) is

      Local_Tmp_1 : driver_types_h.cudaGraphNode_t with
         Address => Node'Address,
         Import;
      Local_Tmp_2 : aliased driver_types_h.cudaMemcpy3DParms with
         Address => P_Node_Params'Address,
         Import;
      Local_Tmp_0 : driver_types_h.cudaError_t  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/cuda_runtime_api.h:8644
      := cuda_runtime_api_h.cudaGraphMemcpyNodeGetParams (Local_Tmp_1, Local_Tmp_2'Access);

   begin
      if Local_Tmp_0 /= 0 then
         Ada.Exceptions.Raise_Exception (CUDA.Exception_Registry.Element (Integer (Local_Tmp_0)));
      end if;
   end Graph_Memcpy_Node_Get_Params;

   procedure Graph_Memcpy_Node_Set_Params (Node : CUDA.Driver_Types.Graph_Node_T; P_Node_Params : out CUDA.Driver_Types.Memcpy3_DParms) is

      Local_Tmp_1 : driver_types_h.cudaGraphNode_t with
         Address => Node'Address,
         Import;
      Local_Tmp_2 : aliased driver_types_h.cudaMemcpy3DParms with
         Address => P_Node_Params'Address,
         Import;
      Local_Tmp_0 : driver_types_h.cudaError_t  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/cuda_runtime_api.h:8667
      := cuda_runtime_api_h.cudaGraphMemcpyNodeSetParams (Local_Tmp_1, Local_Tmp_2'Access);

   begin
      if Local_Tmp_0 /= 0 then
         Ada.Exceptions.Raise_Exception (CUDA.Exception_Registry.Element (Integer (Local_Tmp_0)));
      end if;
   end Graph_Memcpy_Node_Set_Params;

   procedure Graph_Add_Memset_Node (P_Graph_Node : System.Address; Graph : CUDA.Driver_Types.Graph_T; P_Dependencies : System.Address; Num_Dependencies : CUDA.Crtdefs.Size_T; P_Memset_Params : out CUDA.Driver_Types.Memset_Params) is

      Local_Tmp_1 : System.Address with
         Address => P_Graph_Node'Address,
         Import;
      Local_Tmp_2 : driver_types_h.cudaGraph_t with
         Address => Graph'Address,
         Import;
      Local_Tmp_3 : System.Address with
         Address => P_Dependencies'Address,
         Import;
      Local_Tmp_4 : crtdefs_h.size_t with
         Address => Num_Dependencies'Address,
         Import;
      Local_Tmp_5 : aliased driver_types_h.cudaMemsetParams with
         Address => P_Memset_Params'Address,
         Import;
      Local_Tmp_0 : driver_types_h.cudaError_t  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/cuda_runtime_api.h:8709
      := cuda_runtime_api_h.cudaGraphAddMemsetNode (Local_Tmp_1, Local_Tmp_2, Local_Tmp_3, Local_Tmp_4, Local_Tmp_5'Access);

   begin
      if Local_Tmp_0 /= 0 then
         Ada.Exceptions.Raise_Exception (CUDA.Exception_Registry.Element (Integer (Local_Tmp_0)));
      end if;
   end Graph_Add_Memset_Node;

   procedure Graph_Memset_Node_Get_Params (Node : CUDA.Driver_Types.Graph_Node_T; P_Node_Params : out CUDA.Driver_Types.Memset_Params) is

      Local_Tmp_1 : driver_types_h.cudaGraphNode_t with
         Address => Node'Address,
         Import;
      Local_Tmp_2 : aliased driver_types_h.cudaMemsetParams with
         Address => P_Node_Params'Address,
         Import;
      Local_Tmp_0 : driver_types_h.cudaError_t  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/cuda_runtime_api.h:8732
      := cuda_runtime_api_h.cudaGraphMemsetNodeGetParams (Local_Tmp_1, Local_Tmp_2'Access);

   begin
      if Local_Tmp_0 /= 0 then
         Ada.Exceptions.Raise_Exception (CUDA.Exception_Registry.Element (Integer (Local_Tmp_0)));
      end if;
   end Graph_Memset_Node_Get_Params;

   procedure Graph_Memset_Node_Set_Params (Node : CUDA.Driver_Types.Graph_Node_T; P_Node_Params : out CUDA.Driver_Types.Memset_Params) is

      Local_Tmp_1 : driver_types_h.cudaGraphNode_t with
         Address => Node'Address,
         Import;
      Local_Tmp_2 : aliased driver_types_h.cudaMemsetParams with
         Address => P_Node_Params'Address,
         Import;
      Local_Tmp_0 : driver_types_h.cudaError_t  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/cuda_runtime_api.h:8755
      := cuda_runtime_api_h.cudaGraphMemsetNodeSetParams (Local_Tmp_1, Local_Tmp_2'Access);

   begin
      if Local_Tmp_0 /= 0 then
         Ada.Exceptions.Raise_Exception (CUDA.Exception_Registry.Element (Integer (Local_Tmp_0)));
      end if;
   end Graph_Memset_Node_Set_Params;

   procedure Graph_Add_Host_Node (P_Graph_Node : System.Address; Graph : CUDA.Driver_Types.Graph_T; P_Dependencies : System.Address; Num_Dependencies : CUDA.Crtdefs.Size_T; P_Node_Params : out CUDA.Driver_Types.Host_Node_Params) is

      Local_Tmp_1 : System.Address with
         Address => P_Graph_Node'Address,
         Import;
      Local_Tmp_2 : driver_types_h.cudaGraph_t with
         Address => Graph'Address,
         Import;
      Local_Tmp_3 : System.Address with
         Address => P_Dependencies'Address,
         Import;
      Local_Tmp_4 : crtdefs_h.size_t with
         Address => Num_Dependencies'Address,
         Import;
      Local_Tmp_5 : aliased driver_types_h.cudaHostNodeParams with
         Address => P_Node_Params'Address,
         Import;
      Local_Tmp_0 : driver_types_h.cudaError_t  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/cuda_runtime_api.h:8796
      := cuda_runtime_api_h.cudaGraphAddHostNode (Local_Tmp_1, Local_Tmp_2, Local_Tmp_3, Local_Tmp_4, Local_Tmp_5'Access);

   begin
      if Local_Tmp_0 /= 0 then
         Ada.Exceptions.Raise_Exception (CUDA.Exception_Registry.Element (Integer (Local_Tmp_0)));
      end if;
   end Graph_Add_Host_Node;

   procedure Graph_Host_Node_Get_Params (Node : CUDA.Driver_Types.Graph_Node_T; P_Node_Params : out CUDA.Driver_Types.Host_Node_Params) is

      Local_Tmp_1 : driver_types_h.cudaGraphNode_t with
         Address => Node'Address,
         Import;
      Local_Tmp_2 : aliased driver_types_h.cudaHostNodeParams with
         Address => P_Node_Params'Address,
         Import;
      Local_Tmp_0 : driver_types_h.cudaError_t  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/cuda_runtime_api.h:8819
      := cuda_runtime_api_h.cudaGraphHostNodeGetParams (Local_Tmp_1, Local_Tmp_2'Access);

   begin
      if Local_Tmp_0 /= 0 then
         Ada.Exceptions.Raise_Exception (CUDA.Exception_Registry.Element (Integer (Local_Tmp_0)));
      end if;
   end Graph_Host_Node_Get_Params;

   procedure Graph_Host_Node_Set_Params (Node : CUDA.Driver_Types.Graph_Node_T; P_Node_Params : out CUDA.Driver_Types.Host_Node_Params) is

      Local_Tmp_1 : driver_types_h.cudaGraphNode_t with
         Address => Node'Address,
         Import;
      Local_Tmp_2 : aliased driver_types_h.cudaHostNodeParams with
         Address => P_Node_Params'Address,
         Import;
      Local_Tmp_0 : driver_types_h.cudaError_t  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/cuda_runtime_api.h:8842
      := cuda_runtime_api_h.cudaGraphHostNodeSetParams (Local_Tmp_1, Local_Tmp_2'Access);

   begin
      if Local_Tmp_0 /= 0 then
         Ada.Exceptions.Raise_Exception (CUDA.Exception_Registry.Element (Integer (Local_Tmp_0)));
      end if;
   end Graph_Host_Node_Set_Params;

   procedure Graph_Add_Child_Graph_Node (P_Graph_Node : System.Address; Graph : CUDA.Driver_Types.Graph_T; P_Dependencies : System.Address; Num_Dependencies : CUDA.Crtdefs.Size_T; Child_Graph : CUDA.Driver_Types.Graph_T) is

      Local_Tmp_1 : System.Address with
         Address => P_Graph_Node'Address,
         Import;
      Local_Tmp_2 : driver_types_h.cudaGraph_t with
         Address => Graph'Address,
         Import;
      Local_Tmp_3 : System.Address with
         Address => P_Dependencies'Address,
         Import;
      Local_Tmp_4 : crtdefs_h.size_t with
         Address => Num_Dependencies'Address,
         Import;
      Local_Tmp_5 : driver_types_h.cudaGraph_t with
         Address => Child_Graph'Address,
         Import;
      Local_Tmp_0 : driver_types_h.cudaError_t  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/cuda_runtime_api.h:8880
      := cuda_runtime_api_h.cudaGraphAddChildGraphNode (Local_Tmp_1, Local_Tmp_2, Local_Tmp_3, Local_Tmp_4, Local_Tmp_5);

   begin
      if Local_Tmp_0 /= 0 then
         Ada.Exceptions.Raise_Exception (CUDA.Exception_Registry.Element (Integer (Local_Tmp_0)));
      end if;
   end Graph_Add_Child_Graph_Node;

   procedure Graph_Child_Graph_Node_Get_Graph (Node : CUDA.Driver_Types.Graph_Node_T; P_Graph : System.Address) is

      Local_Tmp_1 : driver_types_h.cudaGraphNode_t with
         Address => Node'Address,
         Import;
      Local_Tmp_2 : System.Address with
         Address => P_Graph'Address,
         Import;
      Local_Tmp_0 : driver_types_h.cudaError_t  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/cuda_runtime_api.h:8904
      := cuda_runtime_api_h.cudaGraphChildGraphNodeGetGraph (Local_Tmp_1, Local_Tmp_2);

   begin
      if Local_Tmp_0 /= 0 then
         Ada.Exceptions.Raise_Exception (CUDA.Exception_Registry.Element (Integer (Local_Tmp_0)));
      end if;
   end Graph_Child_Graph_Node_Get_Graph;

   procedure Graph_Add_Empty_Node (P_Graph_Node : System.Address; Graph : CUDA.Driver_Types.Graph_T; P_Dependencies : System.Address; Num_Dependencies : CUDA.Crtdefs.Size_T) is

      Local_Tmp_1 : System.Address with
         Address => P_Graph_Node'Address,
         Import;
      Local_Tmp_2 : driver_types_h.cudaGraph_t with
         Address => Graph'Address,
         Import;
      Local_Tmp_3 : System.Address with
         Address => P_Dependencies'Address,
         Import;
      Local_Tmp_4 : crtdefs_h.size_t with
         Address => Num_Dependencies'Address,
         Import;
      Local_Tmp_0 : driver_types_h.cudaError_t  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/cuda_runtime_api.h:8941
      := cuda_runtime_api_h.cudaGraphAddEmptyNode (Local_Tmp_1, Local_Tmp_2, Local_Tmp_3, Local_Tmp_4);

   begin
      if Local_Tmp_0 /= 0 then
         Ada.Exceptions.Raise_Exception (CUDA.Exception_Registry.Element (Integer (Local_Tmp_0)));
      end if;
   end Graph_Add_Empty_Node;

   procedure Graph_Clone (P_Graph_Clone : System.Address; Original_Graph : CUDA.Driver_Types.Graph_T) is

      Local_Tmp_1 : System.Address with
         Address => P_Graph_Clone'Address,
         Import;
      Local_Tmp_2 : driver_types_h.cudaGraph_t with
         Address => Original_Graph'Address,
         Import;
      Local_Tmp_0 : driver_types_h.cudaError_t  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/cuda_runtime_api.h:8968
      := cuda_runtime_api_h.cudaGraphClone (Local_Tmp_1, Local_Tmp_2);

   begin
      if Local_Tmp_0 /= 0 then
         Ada.Exceptions.Raise_Exception (CUDA.Exception_Registry.Element (Integer (Local_Tmp_0)));
      end if;
   end Graph_Clone;

   procedure Graph_Node_Find_In_Clone (P_Node : System.Address; Original_Node : CUDA.Driver_Types.Graph_Node_T; Cloned_Graph : CUDA.Driver_Types.Graph_T) is

      Local_Tmp_1 : System.Address with
         Address => P_Node'Address,
         Import;
      Local_Tmp_2 : driver_types_h.cudaGraphNode_t with
         Address => Original_Node'Address,
         Import;
      Local_Tmp_3 : driver_types_h.cudaGraph_t with
         Address => Cloned_Graph'Address,
         Import;
      Local_Tmp_0 : driver_types_h.cudaError_t  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/cuda_runtime_api.h:8996
      := cuda_runtime_api_h.cudaGraphNodeFindInClone (Local_Tmp_1, Local_Tmp_2, Local_Tmp_3);

   begin
      if Local_Tmp_0 /= 0 then
         Ada.Exceptions.Raise_Exception (CUDA.Exception_Registry.Element (Integer (Local_Tmp_0)));
      end if;
   end Graph_Node_Find_In_Clone;

   procedure Graph_Node_Get_Type (Node : CUDA.Driver_Types.Graph_Node_T; P_Type : out CUDA.Driver_Types.Graph_Node_Type) is

      Local_Tmp_1 : driver_types_h.cudaGraphNode_t with
         Address => Node'Address,
         Import;
      Local_Tmp_2 : aliased driver_types_h.cudaGraphNodeType with
         Address => P_Type'Address,
         Import;
      Local_Tmp_0 : driver_types_h.cudaError_t  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/cuda_runtime_api.h:9027
      := cuda_runtime_api_h.cudaGraphNodeGetType (Local_Tmp_1, Local_Tmp_2'Access);

   begin
      if Local_Tmp_0 /= 0 then
         Ada.Exceptions.Raise_Exception (CUDA.Exception_Registry.Element (Integer (Local_Tmp_0)));
      end if;
   end Graph_Node_Get_Type;

   procedure Graph_Get_Nodes (Graph : CUDA.Driver_Types.Graph_T; Nodes : System.Address; Num_Nodes : out CUDA.Crtdefs.Size_T) is

      Local_Tmp_1 : driver_types_h.cudaGraph_t with
         Address => Graph'Address,
         Import;
      Local_Tmp_2 : System.Address with
         Address => Nodes'Address,
         Import;
      Local_Tmp_3 : aliased crtdefs_h.size_t with
         Address => Num_Nodes'Address,
         Import;
      Local_Tmp_0 : driver_types_h.cudaError_t  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/cuda_runtime_api.h:9058
      := cuda_runtime_api_h.cudaGraphGetNodes (Local_Tmp_1, Local_Tmp_2, Local_Tmp_3'Access);

   begin
      if Local_Tmp_0 /= 0 then
         Ada.Exceptions.Raise_Exception (CUDA.Exception_Registry.Element (Integer (Local_Tmp_0)));
      end if;
   end Graph_Get_Nodes;

   procedure Graph_Get_Root_Nodes (Graph : CUDA.Driver_Types.Graph_T; P_Root_Nodes : System.Address; P_Num_Root_Nodes : out CUDA.Crtdefs.Size_T) is

      Local_Tmp_1 : driver_types_h.cudaGraph_t with
         Address => Graph'Address,
         Import;
      Local_Tmp_2 : System.Address with
         Address => P_Root_Nodes'Address,
         Import;
      Local_Tmp_3 : aliased crtdefs_h.size_t with
         Address => P_Num_Root_Nodes'Address,
         Import;
      Local_Tmp_0 : driver_types_h.cudaError_t  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/cuda_runtime_api.h:9089
      := cuda_runtime_api_h.cudaGraphGetRootNodes (Local_Tmp_1, Local_Tmp_2, Local_Tmp_3'Access);

   begin
      if Local_Tmp_0 /= 0 then
         Ada.Exceptions.Raise_Exception (CUDA.Exception_Registry.Element (Integer (Local_Tmp_0)));
      end if;
   end Graph_Get_Root_Nodes;

   procedure Graph_Get_Edges (Graph : CUDA.Driver_Types.Graph_T; From : System.Address; To : System.Address; Num_Edges : out CUDA.Crtdefs.Size_T) is

      Local_Tmp_1 : driver_types_h.cudaGraph_t with
         Address => Graph'Address,
         Import;
      Local_Tmp_2 : System.Address with
         Address => From'Address,
         Import;
      Local_Tmp_3 : System.Address with
         Address => To'Address,
         Import;
      Local_Tmp_4 : aliased crtdefs_h.size_t with
         Address => Num_Edges'Address,
         Import;
      Local_Tmp_0 : driver_types_h.cudaError_t  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/cuda_runtime_api.h:9123
      := cuda_runtime_api_h.cudaGraphGetEdges (Local_Tmp_1, Local_Tmp_2, Local_Tmp_3, Local_Tmp_4'Access);

   begin
      if Local_Tmp_0 /= 0 then
         Ada.Exceptions.Raise_Exception (CUDA.Exception_Registry.Element (Integer (Local_Tmp_0)));
      end if;
   end Graph_Get_Edges;

   procedure Graph_Node_Get_Dependencies (Node : CUDA.Driver_Types.Graph_Node_T; P_Dependencies : System.Address; P_Num_Dependencies : out CUDA.Crtdefs.Size_T) is

      Local_Tmp_1 : driver_types_h.cudaGraphNode_t with
         Address => Node'Address,
         Import;
      Local_Tmp_2 : System.Address with
         Address => P_Dependencies'Address,
         Import;
      Local_Tmp_3 : aliased crtdefs_h.size_t with
         Address => P_Num_Dependencies'Address,
         Import;
      Local_Tmp_0 : driver_types_h.cudaError_t  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/cuda_runtime_api.h:9154
      := cuda_runtime_api_h.cudaGraphNodeGetDependencies (Local_Tmp_1, Local_Tmp_2, Local_Tmp_3'Access);

   begin
      if Local_Tmp_0 /= 0 then
         Ada.Exceptions.Raise_Exception (CUDA.Exception_Registry.Element (Integer (Local_Tmp_0)));
      end if;
   end Graph_Node_Get_Dependencies;

   procedure Graph_Node_Get_Dependent_Nodes (Node : CUDA.Driver_Types.Graph_Node_T; P_Dependent_Nodes : System.Address; P_Num_Dependent_Nodes : out CUDA.Crtdefs.Size_T) is

      Local_Tmp_1 : driver_types_h.cudaGraphNode_t with
         Address => Node'Address,
         Import;
      Local_Tmp_2 : System.Address with
         Address => P_Dependent_Nodes'Address,
         Import;
      Local_Tmp_3 : aliased crtdefs_h.size_t with
         Address => P_Num_Dependent_Nodes'Address,
         Import;
      Local_Tmp_0 : driver_types_h.cudaError_t  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/cuda_runtime_api.h:9186
      := cuda_runtime_api_h.cudaGraphNodeGetDependentNodes (Local_Tmp_1, Local_Tmp_2, Local_Tmp_3'Access);

   begin
      if Local_Tmp_0 /= 0 then
         Ada.Exceptions.Raise_Exception (CUDA.Exception_Registry.Element (Integer (Local_Tmp_0)));
      end if;
   end Graph_Node_Get_Dependent_Nodes;

   procedure Graph_Add_Dependencies (Graph : CUDA.Driver_Types.Graph_T; From : System.Address; To : System.Address; Num_Dependencies : CUDA.Crtdefs.Size_T) is

      Local_Tmp_1 : driver_types_h.cudaGraph_t with
         Address => Graph'Address,
         Import;
      Local_Tmp_2 : System.Address with
         Address => From'Address,
         Import;
      Local_Tmp_3 : System.Address with
         Address => To'Address,
         Import;
      Local_Tmp_4 : crtdefs_h.size_t with
         Address => Num_Dependencies'Address,
         Import;
      Local_Tmp_0 : driver_types_h.cudaError_t  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/cuda_runtime_api.h:9217
      := cuda_runtime_api_h.cudaGraphAddDependencies (Local_Tmp_1, Local_Tmp_2, Local_Tmp_3, Local_Tmp_4);

   begin
      if Local_Tmp_0 /= 0 then
         Ada.Exceptions.Raise_Exception (CUDA.Exception_Registry.Element (Integer (Local_Tmp_0)));
      end if;
   end Graph_Add_Dependencies;

   procedure Graph_Remove_Dependencies (Graph : CUDA.Driver_Types.Graph_T; From : System.Address; To : System.Address; Num_Dependencies : CUDA.Crtdefs.Size_T) is

      Local_Tmp_1 : driver_types_h.cudaGraph_t with
         Address => Graph'Address,
         Import;
      Local_Tmp_2 : System.Address with
         Address => From'Address,
         Import;
      Local_Tmp_3 : System.Address with
         Address => To'Address,
         Import;
      Local_Tmp_4 : crtdefs_h.size_t with
         Address => Num_Dependencies'Address,
         Import;
      Local_Tmp_0 : driver_types_h.cudaError_t  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/cuda_runtime_api.h:9248
      := cuda_runtime_api_h.cudaGraphRemoveDependencies (Local_Tmp_1, Local_Tmp_2, Local_Tmp_3, Local_Tmp_4);

   begin
      if Local_Tmp_0 /= 0 then
         Ada.Exceptions.Raise_Exception (CUDA.Exception_Registry.Element (Integer (Local_Tmp_0)));
      end if;
   end Graph_Remove_Dependencies;

   procedure Graph_Destroy_Node (Node : CUDA.Driver_Types.Graph_Node_T) is

      Local_Tmp_1 : driver_types_h.cudaGraphNode_t with
         Address => Node'Address,
         Import;
      Local_Tmp_0 : driver_types_h.cudaError_t  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/cuda_runtime_api.h:9274
      := cuda_runtime_api_h.cudaGraphDestroyNode (Local_Tmp_1);

   begin
      if Local_Tmp_0 /= 0 then
         Ada.Exceptions.Raise_Exception (CUDA.Exception_Registry.Element (Integer (Local_Tmp_0)));
      end if;
   end Graph_Destroy_Node;

   procedure Graph_Instantiate (P_Graph_Exec : System.Address; Graph : CUDA.Driver_Types.Graph_T; P_Error_Node : System.Address; P_Log_Buffer : String; Buffer_Size : CUDA.Crtdefs.Size_T) is

      Local_Tmp_1 : System.Address with
         Address => P_Graph_Exec'Address,
         Import;
      Local_Tmp_2 : driver_types_h.cudaGraph_t with
         Address => Graph'Address,
         Import;
      Local_Tmp_3 : System.Address with
         Address => P_Error_Node'Address,
         Import;
      Local_Tmp_4 : Interfaces.C.Strings.chars_ptr := Interfaces.C.Strings.New_String (P_Log_Buffer);
      Local_Tmp_5 : crtdefs_h.size_t with
         Address => Buffer_Size'Address,
         Import;
      Local_Tmp_0 : driver_types_h.cudaError_t  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/cuda_runtime_api.h:9310
      := cuda_runtime_api_h.cudaGraphInstantiate (Local_Tmp_1, Local_Tmp_2, Local_Tmp_3, Local_Tmp_4, Local_Tmp_5);

   begin
      if Local_Tmp_0 /= 0 then
         Ada.Exceptions.Raise_Exception (CUDA.Exception_Registry.Element (Integer (Local_Tmp_0)));
      end if;
      Interfaces.C.Strings.Free (Local_Tmp_4);
   end Graph_Instantiate;

   procedure Graph_Exec_Kernel_Node_Set_Params (H_Graph_Exec : CUDA.Driver_Types.Graph_Exec_T; Node : CUDA.Driver_Types.Graph_Node_T; P_Node_Params : out CUDA.Driver_Types.Kernel_Node_Params) is

      Local_Tmp_1 : driver_types_h.cudaGraphExec_t with
         Address => H_Graph_Exec'Address,
         Import;
      Local_Tmp_2 : driver_types_h.cudaGraphNode_t with
         Address => Node'Address,
         Import;
      Local_Tmp_3 : aliased driver_types_h.cudaKernelNodeParams with
         Address => P_Node_Params'Address,
         Import;
      Local_Tmp_0 : driver_types_h.cudaError_t  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/cuda_runtime_api.h:9344
      := cuda_runtime_api_h.cudaGraphExecKernelNodeSetParams (Local_Tmp_1, Local_Tmp_2, Local_Tmp_3'Access);

   begin
      if Local_Tmp_0 /= 0 then
         Ada.Exceptions.Raise_Exception (CUDA.Exception_Registry.Element (Integer (Local_Tmp_0)));
      end if;
   end Graph_Exec_Kernel_Node_Set_Params;

   procedure Graph_Exec_Memcpy_Node_Set_Params (H_Graph_Exec : CUDA.Driver_Types.Graph_Exec_T; Node : CUDA.Driver_Types.Graph_Node_T; P_Node_Params : out CUDA.Driver_Types.Memcpy3_DParms) is

      Local_Tmp_1 : driver_types_h.cudaGraphExec_t with
         Address => H_Graph_Exec'Address,
         Import;
      Local_Tmp_2 : driver_types_h.cudaGraphNode_t with
         Address => Node'Address,
         Import;
      Local_Tmp_3 : aliased driver_types_h.cudaMemcpy3DParms with
         Address => P_Node_Params'Address,
         Import;
      Local_Tmp_0 : driver_types_h.cudaError_t  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/cuda_runtime_api.h:9385
      := cuda_runtime_api_h.cudaGraphExecMemcpyNodeSetParams (Local_Tmp_1, Local_Tmp_2, Local_Tmp_3'Access);

   begin
      if Local_Tmp_0 /= 0 then
         Ada.Exceptions.Raise_Exception (CUDA.Exception_Registry.Element (Integer (Local_Tmp_0)));
      end if;
   end Graph_Exec_Memcpy_Node_Set_Params;

   procedure Graph_Exec_Memset_Node_Set_Params (H_Graph_Exec : CUDA.Driver_Types.Graph_Exec_T; Node : CUDA.Driver_Types.Graph_Node_T; P_Node_Params : out CUDA.Driver_Types.Memset_Params) is

      Local_Tmp_1 : driver_types_h.cudaGraphExec_t with
         Address => H_Graph_Exec'Address,
         Import;
      Local_Tmp_2 : driver_types_h.cudaGraphNode_t with
         Address => Node'Address,
         Import;
      Local_Tmp_3 : aliased driver_types_h.cudaMemsetParams with
         Address => P_Node_Params'Address,
         Import;
      Local_Tmp_0 : driver_types_h.cudaError_t  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/cuda_runtime_api.h:9426
      := cuda_runtime_api_h.cudaGraphExecMemsetNodeSetParams (Local_Tmp_1, Local_Tmp_2, Local_Tmp_3'Access);

   begin
      if Local_Tmp_0 /= 0 then
         Ada.Exceptions.Raise_Exception (CUDA.Exception_Registry.Element (Integer (Local_Tmp_0)));
      end if;
   end Graph_Exec_Memset_Node_Set_Params;

   procedure Graph_Exec_Host_Node_Set_Params (H_Graph_Exec : CUDA.Driver_Types.Graph_Exec_T; Node : CUDA.Driver_Types.Graph_Node_T; P_Node_Params : out CUDA.Driver_Types.Host_Node_Params) is

      Local_Tmp_1 : driver_types_h.cudaGraphExec_t with
         Address => H_Graph_Exec'Address,
         Import;
      Local_Tmp_2 : driver_types_h.cudaGraphNode_t with
         Address => Node'Address,
         Import;
      Local_Tmp_3 : aliased driver_types_h.cudaHostNodeParams with
         Address => P_Node_Params'Address,
         Import;
      Local_Tmp_0 : driver_types_h.cudaError_t  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/cuda_runtime_api.h:9459
      := cuda_runtime_api_h.cudaGraphExecHostNodeSetParams (Local_Tmp_1, Local_Tmp_2, Local_Tmp_3'Access);

   begin
      if Local_Tmp_0 /= 0 then
         Ada.Exceptions.Raise_Exception (CUDA.Exception_Registry.Element (Integer (Local_Tmp_0)));
      end if;
   end Graph_Exec_Host_Node_Set_Params;

   procedure Graph_Exec_Update (H_Graph_Exec : CUDA.Driver_Types.Graph_Exec_T; H_Graph : CUDA.Driver_Types.Graph_T; H_Error_Node_Out : System.Address; Update_Result_Out : out CUDA.Driver_Types.Graph_Exec_Update_Result) is

      Local_Tmp_1 : driver_types_h.cudaGraphExec_t with
         Address => H_Graph_Exec'Address,
         Import;
      Local_Tmp_2 : driver_types_h.cudaGraph_t with
         Address => H_Graph'Address,
         Import;
      Local_Tmp_3 : System.Address with
         Address => H_Error_Node_Out'Address,
         Import;
      Local_Tmp_4 : aliased driver_types_h.cudaGraphExecUpdateResult with
         Address => Update_Result_Out'Address,
         Import;
      Local_Tmp_0 : driver_types_h.cudaError_t  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/cuda_runtime_api.h:9534
      := cuda_runtime_api_h.cudaGraphExecUpdate (Local_Tmp_1, Local_Tmp_2, Local_Tmp_3, Local_Tmp_4'Access);

   begin
      if Local_Tmp_0 /= 0 then
         Ada.Exceptions.Raise_Exception (CUDA.Exception_Registry.Element (Integer (Local_Tmp_0)));
      end if;
   end Graph_Exec_Update;

   procedure Graph_Launch (Graph_Exec : CUDA.Driver_Types.Graph_Exec_T; Stream : CUDA.Driver_Types.Stream_T) is

      Local_Tmp_1 : driver_types_h.cudaGraphExec_t with
         Address => Graph_Exec'Address,
         Import;
      Local_Tmp_2 : driver_types_h.cudaStream_t with
         Address => Stream'Address,
         Import;
      Local_Tmp_0 : driver_types_h.cudaError_t  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/cuda_runtime_api.h:9559
      := cuda_runtime_api_h.cudaGraphLaunch (Local_Tmp_1, Local_Tmp_2);

   begin
      if Local_Tmp_0 /= 0 then
         Ada.Exceptions.Raise_Exception (CUDA.Exception_Registry.Element (Integer (Local_Tmp_0)));
      end if;
   end Graph_Launch;

   procedure Graph_Exec_Destroy (Graph_Exec : CUDA.Driver_Types.Graph_Exec_T) is

      Local_Tmp_1 : driver_types_h.cudaGraphExec_t with
         Address => Graph_Exec'Address,
         Import;
      Local_Tmp_0 : driver_types_h.cudaError_t  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/cuda_runtime_api.h:9580
      := cuda_runtime_api_h.cudaGraphExecDestroy (Local_Tmp_1);

   begin
      if Local_Tmp_0 /= 0 then
         Ada.Exceptions.Raise_Exception (CUDA.Exception_Registry.Element (Integer (Local_Tmp_0)));
      end if;
   end Graph_Exec_Destroy;

   procedure Graph_Destroy (Graph : CUDA.Driver_Types.Graph_T) is

      Local_Tmp_1 : driver_types_h.cudaGraph_t with
         Address => Graph'Address,
         Import;
      Local_Tmp_0 : driver_types_h.cudaError_t  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/cuda_runtime_api.h:9600
      := cuda_runtime_api_h.cudaGraphDestroy (Local_Tmp_1);

   begin
      if Local_Tmp_0 /= 0 then
         Ada.Exceptions.Raise_Exception (CUDA.Exception_Registry.Element (Integer (Local_Tmp_0)));
      end if;
   end Graph_Destroy;

   procedure Get_Export_Table (Pp_Export_Table : System.Address; P_Export_Table_Id : out CUDA.Driver_Types.UUID_T) is

      Local_Tmp_1 : System.Address with
         Address => Pp_Export_Table'Address,
         Import;
      Local_Tmp_2 : aliased driver_types_h.cudaUUID_t with
         Address => P_Export_Table_Id'Address,
         Import;
      Local_Tmp_0 : driver_types_h.cudaError_t  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/cuda_runtime_api.h:9605
      := cuda_runtime_api_h.cudaGetExportTable (Local_Tmp_1, Local_Tmp_2'Access);

   begin
      if Local_Tmp_0 /= 0 then
         Ada.Exceptions.Raise_Exception (CUDA.Exception_Registry.Element (Integer (Local_Tmp_0)));
      end if;
   end Get_Export_Table;

begin
   null;
end CUDA.Runtime_Api;
