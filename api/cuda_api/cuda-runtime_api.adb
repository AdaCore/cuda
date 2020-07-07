with cuda_runtime_api_h; use cuda_runtime_api_h;

package body CUDA.Runtime_Api is

   function Grid_Dim return CUDA.Vector_Types.Dim3 is

      function Nctaid_X return Interfaces.C.unsigned with
         Inline,
         Import,
         Convention    => C,
         External_Name => "*llvm.nvvm.read.ptx.sreg.nctaid.x";
      function Nctaid_Y return Interfaces.C.unsigned with
         Inline,
         Import,
         Convention    => C,
         External_Name => "*llvm.nvvm.read.ptx.sreg.nctaid.y";
      function Nctaid_Z return Interfaces.C.unsigned with
         Inline,
         Import,
         Convention    => C,
         External_Name => "*llvm.nvvm.read.ptx.sreg.nctaid.z";

   begin
      return (Nctaid_X, Nctaid_Y, Nctaid_Z);
   end Grid_Dim;

   function Block_Idx return CUDA.Vector_Types.Uint3 is

      function Ctaid_X return Interfaces.C.unsigned with
         Inline,
         Import,
         Convention    => C,
         External_Name => "*llvm.nvvm.read.ptx.sreg.ctaid.x";
      function Ctaid_Y return Interfaces.C.unsigned with
         Inline,
         Import,
         Convention    => C,
         External_Name => "*llvm.nvvm.read.ptx.sreg.ctaid.y";
      function Ctaid_Z return Interfaces.C.unsigned with
         Inline,
         Import,
         Convention    => C,
         External_Name => "*llvm.nvvm.read.ptx.sreg.ctaid.z";

   begin
      return (Ctaid_X, Ctaid_Y, Ctaid_Z);
   end Block_Idx;

   function Block_Dim return CUDA.Vector_Types.Dim3 is

      function Ntid_X return Interfaces.C.unsigned with
         Inline,
         Import,
         Convention    => C,
         External_Name => "*llvm.nvvm.read.ptx.sreg.ntid.x";
      function Ntid_Y return Interfaces.C.unsigned with
         Inline,
         Import,
         Convention    => C,
         External_Name => "*llvm.nvvm.read.ptx.sreg.ntid.y";
      function Ntid_Z return Interfaces.C.unsigned with
         Inline,
         Import,
         Convention    => C,
         External_Name => "*llvm.nvvm.read.ptx.sreg.ntid.z";

   begin
      return (Ntid_X, Ntid_Y, Ntid_Z);
   end Block_Dim;

   function Thread_Idx return CUDA.Vector_Types.Uint3 is

      function Tid_X return Interfaces.C.unsigned with
         Inline,
         Import,
         Convention    => C,
         External_Name => "*llvm.nvvm.read.ptx.sreg.tid.x";
      function Tid_Y return Interfaces.C.unsigned with
         Inline,
         Import,
         Convention    => C,
         External_Name => "*llvm.nvvm.read.ptx.sreg.tid.y";
      function Tid_Z return Interfaces.C.unsigned with
         Inline,
         Import,
         Convention    => C,
         External_Name => "*llvm.nvvm.read.ptx.sreg.tid.z";

   begin
      return (Tid_X, Tid_Y, Tid_Z);
   end Thread_Idx;

   function Wrap_Size return Interfaces.C.int is

      function Wrapsize return Interfaces.C.int with
         Inline,
         Import,
         Convention    => C,
         External_Name => "*llvm.nvvm.read.ptx.sreg.wrapsize";

   begin
      return Wrapsize;
   end Wrap_Size;

   procedure Device_Reset is
   begin
      declare

         Temp_res_1 : Integer := Integer (cuda_runtime_api_h.cudaDeviceReset);

      begin
         null;

         declare
         begin
            if Temp_res_1 /= 0 then
               Ada.Exceptions.Raise_Exception
                 (CUDA.Exception_Registry.Element (Integer (Temp_res_1)));
            end if;
         end;
      end;
   end Device_Reset;

   procedure Device_Synchronize is
   begin
      declare

         Temp_res_1 : Integer :=
           Integer (cuda_runtime_api_h.cudaDeviceSynchronize);

      begin
         null;

         declare
         begin
            if Temp_res_1 /= 0 then
               Ada.Exceptions.Raise_Exception
                 (CUDA.Exception_Registry.Element (Integer (Temp_res_1)));
            end if;
         end;
      end;
   end Device_Synchronize;

   procedure Device_Set_Limit
     (Limit : CUDA.Driver_Types.Limit; Value : CUDA.Corecrt.Size_T)
   is

      Temp_local_3 : aliased driver_types_h.cudaLimit with
         Address => Limit'Address,
         Import;
      Temp_local_5 : aliased corecrt_h.size_t with
         Address => Value'Address,
         Import;

   begin
      declare

         Temp_res_1 : Integer :=
           Integer
             (cuda_runtime_api_h.cudaDeviceSetLimit
                (Temp_local_3, Temp_local_5));

      begin
         null;

         declare
         begin
            if Temp_res_1 /= 0 then
               Ada.Exceptions.Raise_Exception
                 (CUDA.Exception_Registry.Element (Integer (Temp_res_1)));
            end if;
         end;
      end;
   end Device_Set_Limit;

   function Device_Get_Limit
     (Limit : CUDA.Driver_Types.Limit) return CUDA.Corecrt.Size_T
   is

      Temp_call_1 : aliased corecrt_h.size_t;
      Temp_ret_2  : aliased CUDA.Corecrt.Size_T with
         Address => Temp_call_1'Address,
         Import;
      Temp_local_4 : aliased driver_types_h.cudaLimit with
         Address => Limit'Address,
         Import;

   begin
      declare

         Temp_res_1 : Integer :=
           Integer
             (cuda_runtime_api_h.cudaDeviceGetLimit
                (Temp_call_1'Unchecked_Access, Temp_local_4));

      begin
         null;

         declare
         begin
            if Temp_res_1 /= 0 then
               Ada.Exceptions.Raise_Exception
                 (CUDA.Exception_Registry.Element (Integer (Temp_res_1)));
            end if;
            return Temp_ret_2;
         end;
      end;
   end Device_Get_Limit;

   function Device_Get_Cache_Config return CUDA.Driver_Types.Func_Cache is

      Temp_call_1 : aliased driver_types_h.cudaFuncCache;
      Temp_ret_2  : aliased CUDA.Driver_Types.Func_Cache with
         Address => Temp_call_1'Address,
         Import;

   begin
      declare

         Temp_res_1 : Integer :=
           Integer
             (cuda_runtime_api_h.cudaDeviceGetCacheConfig
                (Temp_call_1'Unchecked_Access));

      begin
         null;

         declare
         begin
            if Temp_res_1 /= 0 then
               Ada.Exceptions.Raise_Exception
                 (CUDA.Exception_Registry.Element (Integer (Temp_res_1)));
            end if;
            return Temp_ret_2;
         end;
      end;
   end Device_Get_Cache_Config;

   function Device_Get_Stream_Priority_Range
     (Greatest_Priority : out int) return int
   is

      Temp_call_1 : aliased int;
      Temp_ret_2  : aliased int with
         Address => Temp_call_1'Address,
         Import;
      Temp_call_3 : aliased int with
         Address => Greatest_Priority'Address,
         Import;

   begin
      declare

         Temp_res_1 : Integer :=
           Integer
             (cuda_runtime_api_h.cudaDeviceGetStreamPriorityRange
                (Temp_call_1'Unchecked_Access, Temp_call_3'Unchecked_Access));

      begin
         null;

         declare
         begin
            if Temp_res_1 /= 0 then
               Ada.Exceptions.Raise_Exception
                 (CUDA.Exception_Registry.Element (Integer (Temp_res_1)));
            end if;
            return Temp_ret_2;
         end;
      end;
   end Device_Get_Stream_Priority_Range;

   procedure Device_Set_Cache_Config
     (Cache_Config : CUDA.Driver_Types.Func_Cache)
   is

      Temp_local_3 : aliased driver_types_h.cudaFuncCache with
         Address => Cache_Config'Address,
         Import;

   begin
      declare

         Temp_res_1 : Integer :=
           Integer (cuda_runtime_api_h.cudaDeviceSetCacheConfig (Temp_local_3));

      begin
         null;

         declare
         begin
            if Temp_res_1 /= 0 then
               Ada.Exceptions.Raise_Exception
                 (CUDA.Exception_Registry.Element (Integer (Temp_res_1)));
            end if;
         end;
      end;
   end Device_Set_Cache_Config;

   function Device_Get_Shared_Mem_Config
      return CUDA.Driver_Types.Shared_Mem_Config
   is

      Temp_call_1 : aliased driver_types_h.cudaSharedMemConfig;
      Temp_ret_2  : aliased CUDA.Driver_Types.Shared_Mem_Config with
         Address => Temp_call_1'Address,
         Import;

   begin
      declare

         Temp_res_1 : Integer :=
           Integer
             (cuda_runtime_api_h.cudaDeviceGetSharedMemConfig
                (Temp_call_1'Unchecked_Access));

      begin
         null;

         declare
         begin
            if Temp_res_1 /= 0 then
               Ada.Exceptions.Raise_Exception
                 (CUDA.Exception_Registry.Element (Integer (Temp_res_1)));
            end if;
            return Temp_ret_2;
         end;
      end;
   end Device_Get_Shared_Mem_Config;

   procedure Device_Set_Shared_Mem_Config
     (Config : CUDA.Driver_Types.Shared_Mem_Config)
   is

      Temp_local_3 : aliased driver_types_h.cudaSharedMemConfig with
         Address => Config'Address,
         Import;

   begin
      declare

         Temp_res_1 : Integer :=
           Integer
             (cuda_runtime_api_h.cudaDeviceSetSharedMemConfig (Temp_local_3));

      begin
         null;

         declare
         begin
            if Temp_res_1 /= 0 then
               Ada.Exceptions.Raise_Exception
                 (CUDA.Exception_Registry.Element (Integer (Temp_res_1)));
            end if;
         end;
      end;
   end Device_Set_Shared_Mem_Config;

   function Device_Get_By_PCIBus_Id (Pci_Bus_Id : String) return int is

      Temp_call_1 : aliased int;
      Temp_ret_2  : aliased int with
         Address => Temp_call_1'Address,
         Import;
      Temp_c_string_3 : Interfaces.C.Strings.chars_ptr :=
        Interfaces.C.Strings.New_String (Pci_Bus_Id);

   begin
      declare

         Temp_res_1 : Integer :=
           Integer
             (cuda_runtime_api_h.cudaDeviceGetByPCIBusId
                (Temp_call_1'Unchecked_Access, Temp_c_string_3));

      begin
         null;

         declare
         begin
            if Temp_res_1 /= 0 then
               Ada.Exceptions.Raise_Exception
                 (CUDA.Exception_Registry.Element (Integer (Temp_res_1)));
            end if;
            Interfaces.C.Strings.Free (Temp_c_string_3);
            return Temp_ret_2;
         end;
      end;
   end Device_Get_By_PCIBus_Id;

   procedure Device_Get_PCIBus_Id (Pci_Bus_Id : String; Len : int; Device : int)
   is

      Temp_c_string_2 : Interfaces.C.Strings.chars_ptr :=
        Interfaces.C.Strings.New_String (Pci_Bus_Id);
      Temp_local_5 : aliased int with
         Address => Len'Address,
         Import;
      Temp_local_7 : aliased int with
         Address => Device'Address,
         Import;

   begin
      declare

         Temp_res_1 : Integer :=
           Integer
             (cuda_runtime_api_h.cudaDeviceGetPCIBusId
                (Temp_c_string_2, Temp_local_5, Temp_local_7));

      begin
         null;

         declare
         begin
            if Temp_res_1 /= 0 then
               Ada.Exceptions.Raise_Exception
                 (CUDA.Exception_Registry.Element (Integer (Temp_res_1)));
            end if;
            Interfaces.C.Strings.Free (Temp_c_string_2);
         end;
      end;
   end Device_Get_PCIBus_Id;

   function Ipc_Get_Event_Handle
     (Event : CUDA.Driver_Types.Event_T)
      return CUDA.Driver_Types.Ipc_Event_Handle_T
   is

      Temp_call_1 : aliased driver_types_h.cudaIpcEventHandle_t;
      Temp_ret_2  : aliased CUDA.Driver_Types.Ipc_Event_Handle_T with
         Address => Temp_call_1'Address,
         Import;
      Temp_local_4 : aliased driver_types_h.cudaEvent_t with
         Address => Event'Address,
         Import;

   begin
      declare

         Temp_res_1 : Integer :=
           Integer
             (cuda_runtime_api_h.cudaIpcGetEventHandle
                (Temp_call_1'Unchecked_Access, Temp_local_4));

      begin
         null;

         declare
         begin
            if Temp_res_1 /= 0 then
               Ada.Exceptions.Raise_Exception
                 (CUDA.Exception_Registry.Element (Integer (Temp_res_1)));
            end if;
            return Temp_ret_2;
         end;
      end;
   end Ipc_Get_Event_Handle;

   function Ipc_Open_Event_Handle
     (Handle : CUDA.Driver_Types.Ipc_Event_Handle_T)
      return CUDA.Driver_Types.Event_T
   is

      Temp_ret_2   : aliased CUDA.Driver_Types.Event_T;
      Temp_call_1  : aliased System.Address := Temp_ret_2'Address;
      Temp_local_4 : aliased driver_types_h.cudaIpcEventHandle_t with
         Address => Handle'Address,
         Import;

   begin
      declare

         Temp_res_1 : Integer :=
           Integer
             (cuda_runtime_api_h.cudaIpcOpenEventHandle
                (Temp_call_1, Temp_local_4));

      begin
         null;

         declare
         begin
            if Temp_res_1 /= 0 then
               Ada.Exceptions.Raise_Exception
                 (CUDA.Exception_Registry.Element (Integer (Temp_res_1)));
            end if;
            return Temp_ret_2;
         end;
      end;
   end Ipc_Open_Event_Handle;

   function Ipc_Get_Mem_Handle
     (Dev_Ptr : System.Address) return CUDA.Driver_Types.Ipc_Mem_Handle_T
   is

      Temp_call_1 : aliased driver_types_h.cudaIpcMemHandle_t;
      Temp_ret_2  : aliased CUDA.Driver_Types.Ipc_Mem_Handle_T with
         Address => Temp_call_1'Address,
         Import;
      Temp_local_4 : aliased System.Address with
         Address => Dev_Ptr'Address,
         Import;

   begin
      declare

         Temp_res_1 : Integer :=
           Integer
             (cuda_runtime_api_h.cudaIpcGetMemHandle
                (Temp_call_1'Unchecked_Access, Temp_local_4));

      begin
         null;

         declare
         begin
            if Temp_res_1 /= 0 then
               Ada.Exceptions.Raise_Exception
                 (CUDA.Exception_Registry.Element (Integer (Temp_res_1)));
            end if;
            return Temp_ret_2;
         end;
      end;
   end Ipc_Get_Mem_Handle;

   procedure Ipc_Open_Mem_Handle
     (Dev_Ptr : System.Address; Handle : CUDA.Driver_Types.Ipc_Mem_Handle_T;
      Flags   : unsigned)
   is

      Temp_local_3 : aliased System.Address with
         Address => Dev_Ptr'Address,
         Import;
      Temp_local_5 : aliased driver_types_h.cudaIpcMemHandle_t with
         Address => Handle'Address,
         Import;
      Temp_local_7 : aliased unsigned with
         Address => Flags'Address,
         Import;

   begin
      declare

         Temp_res_1 : Integer :=
           Integer
             (cuda_runtime_api_h.cudaIpcOpenMemHandle
                (Temp_local_3, Temp_local_5, Temp_local_7));

      begin
         null;

         declare
         begin
            if Temp_res_1 /= 0 then
               Ada.Exceptions.Raise_Exception
                 (CUDA.Exception_Registry.Element (Integer (Temp_res_1)));
            end if;
         end;
      end;
   end Ipc_Open_Mem_Handle;

   procedure Ipc_Close_Mem_Handle (Dev_Ptr : System.Address) is

      Temp_local_3 : aliased System.Address with
         Address => Dev_Ptr'Address,
         Import;

   begin
      declare

         Temp_res_1 : Integer :=
           Integer (cuda_runtime_api_h.cudaIpcCloseMemHandle (Temp_local_3));

      begin
         null;

         declare
         begin
            if Temp_res_1 /= 0 then
               Ada.Exceptions.Raise_Exception
                 (CUDA.Exception_Registry.Element (Integer (Temp_res_1)));
            end if;
         end;
      end;
   end Ipc_Close_Mem_Handle;

   procedure Thread_Exit is
   begin
      declare

         Temp_res_1 : Integer := Integer (cuda_runtime_api_h.cudaThreadExit);

      begin
         null;

         declare
         begin
            if Temp_res_1 /= 0 then
               Ada.Exceptions.Raise_Exception
                 (CUDA.Exception_Registry.Element (Integer (Temp_res_1)));
            end if;
         end;
      end;
   end Thread_Exit;

   procedure Thread_Synchronize is
   begin
      declare

         Temp_res_1 : Integer :=
           Integer (cuda_runtime_api_h.cudaThreadSynchronize);

      begin
         null;

         declare
         begin
            if Temp_res_1 /= 0 then
               Ada.Exceptions.Raise_Exception
                 (CUDA.Exception_Registry.Element (Integer (Temp_res_1)));
            end if;
         end;
      end;
   end Thread_Synchronize;

   procedure Thread_Set_Limit
     (Limit : CUDA.Driver_Types.Limit; Value : CUDA.Corecrt.Size_T)
   is

      Temp_local_3 : aliased driver_types_h.cudaLimit with
         Address => Limit'Address,
         Import;
      Temp_local_5 : aliased corecrt_h.size_t with
         Address => Value'Address,
         Import;

   begin
      declare

         Temp_res_1 : Integer :=
           Integer
             (cuda_runtime_api_h.cudaThreadSetLimit
                (Temp_local_3, Temp_local_5));

      begin
         null;

         declare
         begin
            if Temp_res_1 /= 0 then
               Ada.Exceptions.Raise_Exception
                 (CUDA.Exception_Registry.Element (Integer (Temp_res_1)));
            end if;
         end;
      end;
   end Thread_Set_Limit;

   function Thread_Get_Limit
     (Limit : CUDA.Driver_Types.Limit) return CUDA.Corecrt.Size_T
   is

      Temp_call_1 : aliased corecrt_h.size_t;
      Temp_ret_2  : aliased CUDA.Corecrt.Size_T with
         Address => Temp_call_1'Address,
         Import;
      Temp_local_4 : aliased driver_types_h.cudaLimit with
         Address => Limit'Address,
         Import;

   begin
      declare

         Temp_res_1 : Integer :=
           Integer
             (cuda_runtime_api_h.cudaThreadGetLimit
                (Temp_call_1'Unchecked_Access, Temp_local_4));

      begin
         null;

         declare
         begin
            if Temp_res_1 /= 0 then
               Ada.Exceptions.Raise_Exception
                 (CUDA.Exception_Registry.Element (Integer (Temp_res_1)));
            end if;
            return Temp_ret_2;
         end;
      end;
   end Thread_Get_Limit;

   function Thread_Get_Cache_Config return CUDA.Driver_Types.Func_Cache is

      Temp_call_1 : aliased driver_types_h.cudaFuncCache;
      Temp_ret_2  : aliased CUDA.Driver_Types.Func_Cache with
         Address => Temp_call_1'Address,
         Import;

   begin
      declare

         Temp_res_1 : Integer :=
           Integer
             (cuda_runtime_api_h.cudaThreadGetCacheConfig
                (Temp_call_1'Unchecked_Access));

      begin
         null;

         declare
         begin
            if Temp_res_1 /= 0 then
               Ada.Exceptions.Raise_Exception
                 (CUDA.Exception_Registry.Element (Integer (Temp_res_1)));
            end if;
            return Temp_ret_2;
         end;
      end;
   end Thread_Get_Cache_Config;

   procedure Thread_Set_Cache_Config
     (Cache_Config : CUDA.Driver_Types.Func_Cache)
   is

      Temp_local_3 : aliased driver_types_h.cudaFuncCache with
         Address => Cache_Config'Address,
         Import;

   begin
      declare

         Temp_res_1 : Integer :=
           Integer (cuda_runtime_api_h.cudaThreadSetCacheConfig (Temp_local_3));

      begin
         null;

         declare
         begin
            if Temp_res_1 /= 0 then
               Ada.Exceptions.Raise_Exception
                 (CUDA.Exception_Registry.Element (Integer (Temp_res_1)));
            end if;
         end;
      end;
   end Thread_Set_Cache_Config;

   function Get_Last_Error return CUDA.Driver_Types.Error_T is
   begin
      declare

         Temp_result_orig_1 : aliased driver_types_h.cudaError_t :=
           cuda_runtime_api_h.cudaGetLastError;

      begin
         null;

         declare

            Temp_result_wrapped_2 : aliased CUDA.Driver_Types.Error_T with
               Address => Temp_result_orig_1'Address,
               Import;

         begin
            return Temp_result_wrapped_2;
         end;
      end;
   end Get_Last_Error;

   function Peek_At_Last_Error return CUDA.Driver_Types.Error_T is
   begin
      declare

         Temp_result_orig_1 : aliased driver_types_h.cudaError_t :=
           cuda_runtime_api_h.cudaPeekAtLastError;

      begin
         null;

         declare

            Temp_result_wrapped_2 : aliased CUDA.Driver_Types.Error_T with
               Address => Temp_result_orig_1'Address,
               Import;

         begin
            return Temp_result_wrapped_2;
         end;
      end;
   end Peek_At_Last_Error;

   function Get_Error_Name (Arg1 : CUDA.Driver_Types.Error_T) return String is

      Temp_local_3 : aliased driver_types_h.cudaError_t with
         Address => Arg1'Address,
         Import;

   begin
      declare

         Temp_c_string_1 : aliased Interfaces.C.Strings.chars_ptr :=
           cuda_runtime_api_h.cudaGetErrorName (Temp_local_3);

      begin
         null;

         declare
         begin
            return Interfaces.C.Strings.Value (Temp_c_string_1);
         end;
      end;
   end Get_Error_Name;

   function Get_Error_String (Arg1 : CUDA.Driver_Types.Error_T) return String is

      Temp_local_3 : aliased driver_types_h.cudaError_t with
         Address => Arg1'Address,
         Import;

   begin
      declare

         Temp_c_string_1 : aliased Interfaces.C.Strings.chars_ptr :=
           cuda_runtime_api_h.cudaGetErrorString (Temp_local_3);

      begin
         null;

         declare
         begin
            return Interfaces.C.Strings.Value (Temp_c_string_1);
         end;
      end;
   end Get_Error_String;

   function Get_Device_Count return int is

      Temp_call_1 : aliased int;
      Temp_ret_2  : aliased int with
         Address => Temp_call_1'Address,
         Import;

   begin
      declare

         Temp_res_1 : Integer :=
           Integer
             (cuda_runtime_api_h.cudaGetDeviceCount
                (Temp_call_1'Unchecked_Access));

      begin
         null;

         declare
         begin
            if Temp_res_1 /= 0 then
               Ada.Exceptions.Raise_Exception
                 (CUDA.Exception_Registry.Element (Integer (Temp_res_1)));
            end if;
            return Temp_ret_2;
         end;
      end;
   end Get_Device_Count;

   function Get_Device_Properties
     (Device : int) return CUDA.Driver_Types.Device_Prop
   is

      Temp_call_1 : aliased driver_types_h.cudaDeviceProp;
      Temp_ret_2  : aliased CUDA.Driver_Types.Device_Prop with
         Address => Temp_call_1'Address,
         Import;
      Temp_local_4 : aliased int with
         Address => Device'Address,
         Import;

   begin
      declare

         Temp_res_1 : Integer :=
           Integer
             (cuda_runtime_api_h.cudaGetDeviceProperties
                (Temp_call_1'Unchecked_Access, Temp_local_4));

      begin
         null;

         declare
         begin
            if Temp_res_1 /= 0 then
               Ada.Exceptions.Raise_Exception
                 (CUDA.Exception_Registry.Element (Integer (Temp_res_1)));
            end if;
            return Temp_ret_2;
         end;
      end;
   end Get_Device_Properties;

   function Device_Get_Attribute
     (Attr : CUDA.Driver_Types.Device_Attr; Device : int) return int
   is

      Temp_call_1 : aliased int;
      Temp_ret_2  : aliased int with
         Address => Temp_call_1'Address,
         Import;
      Temp_local_4 : aliased driver_types_h.cudaDeviceAttr with
         Address => Attr'Address,
         Import;
      Temp_local_6 : aliased int with
         Address => Device'Address,
         Import;

   begin
      declare

         Temp_res_1 : Integer :=
           Integer
             (cuda_runtime_api_h.cudaDeviceGetAttribute
                (Temp_call_1'Unchecked_Access, Temp_local_4, Temp_local_6));

      begin
         null;

         declare
         begin
            if Temp_res_1 /= 0 then
               Ada.Exceptions.Raise_Exception
                 (CUDA.Exception_Registry.Element (Integer (Temp_res_1)));
            end if;
            return Temp_ret_2;
         end;
      end;
   end Device_Get_Attribute;

   procedure Device_Get_Nv_Sci_Sync_Attributes
     (Nv_Sci_Sync_Attr_List : System.Address; Device : int; Flags : int)
   is

      Temp_local_3 : aliased System.Address with
         Address => Nv_Sci_Sync_Attr_List'Address,
         Import;
      Temp_local_5 : aliased int with
         Address => Device'Address,
         Import;
      Temp_local_7 : aliased int with
         Address => Flags'Address,
         Import;

   begin
      declare

         Temp_res_1 : Integer :=
           Integer
             (cuda_runtime_api_h.cudaDeviceGetNvSciSyncAttributes
                (Temp_local_3, Temp_local_5, Temp_local_7));

      begin
         null;

         declare
         begin
            if Temp_res_1 /= 0 then
               Ada.Exceptions.Raise_Exception
                 (CUDA.Exception_Registry.Element (Integer (Temp_res_1)));
            end if;
         end;
      end;
   end Device_Get_Nv_Sci_Sync_Attributes;

   function Device_Get_P2_PAttribute
     (Attr       : CUDA.Driver_Types.Device_P2_PAttr; Src_Device : int;
      Dst_Device : int) return int
   is

      Temp_call_1 : aliased int;
      Temp_ret_2  : aliased int with
         Address => Temp_call_1'Address,
         Import;
      Temp_local_4 : aliased driver_types_h.cudaDeviceP2PAttr with
         Address => Attr'Address,
         Import;
      Temp_local_6 : aliased int with
         Address => Src_Device'Address,
         Import;
      Temp_local_8 : aliased int with
         Address => Dst_Device'Address,
         Import;

   begin
      declare

         Temp_res_1 : Integer :=
           Integer
             (cuda_runtime_api_h.cudaDeviceGetP2PAttribute
                (Temp_call_1'Unchecked_Access, Temp_local_4, Temp_local_6,
                 Temp_local_8));

      begin
         null;

         declare
         begin
            if Temp_res_1 /= 0 then
               Ada.Exceptions.Raise_Exception
                 (CUDA.Exception_Registry.Element (Integer (Temp_res_1)));
            end if;
            return Temp_ret_2;
         end;
      end;
   end Device_Get_P2_PAttribute;

   procedure Choose_Device
     (Device : out int; Prop : out CUDA.Driver_Types.Device_Prop)
   is

      Temp_call_2 : aliased int with
         Address => Device'Address,
         Import;
      Temp_call_4 : aliased driver_types_h.cudaDeviceProp with
         Address => Prop'Address,
         Import;

   begin
      declare

         Temp_res_1 : Integer :=
           Integer
             (cuda_runtime_api_h.cudaChooseDevice
                (Temp_call_2'Unchecked_Access, Temp_call_4'Unchecked_Access));

      begin
         null;

         declare
         begin
            if Temp_res_1 /= 0 then
               Ada.Exceptions.Raise_Exception
                 (CUDA.Exception_Registry.Element (Integer (Temp_res_1)));
            end if;
         end;
      end;
   end Choose_Device;

   procedure Set_Device (Device : int) is

      Temp_local_3 : aliased int with
         Address => Device'Address,
         Import;

   begin
      declare

         Temp_res_1 : Integer :=
           Integer (cuda_runtime_api_h.cudaSetDevice (Temp_local_3));

      begin
         null;

         declare
         begin
            if Temp_res_1 /= 0 then
               Ada.Exceptions.Raise_Exception
                 (CUDA.Exception_Registry.Element (Integer (Temp_res_1)));
            end if;
         end;
      end;
   end Set_Device;

   function Get_Device return int is

      Temp_call_1 : aliased int;
      Temp_ret_2  : aliased int with
         Address => Temp_call_1'Address,
         Import;

   begin
      declare

         Temp_res_1 : Integer :=
           Integer
             (cuda_runtime_api_h.cudaGetDevice (Temp_call_1'Unchecked_Access));

      begin
         null;

         declare
         begin
            if Temp_res_1 /= 0 then
               Ada.Exceptions.Raise_Exception
                 (CUDA.Exception_Registry.Element (Integer (Temp_res_1)));
            end if;
            return Temp_ret_2;
         end;
      end;
   end Get_Device;

   procedure Set_Valid_Devices (Device_Arr : out int; Len : int) is

      Temp_call_2 : aliased int with
         Address => Device_Arr'Address,
         Import;
      Temp_local_5 : aliased int with
         Address => Len'Address,
         Import;

   begin
      declare

         Temp_res_1 : Integer :=
           Integer
             (cuda_runtime_api_h.cudaSetValidDevices
                (Temp_call_2'Unchecked_Access, Temp_local_5));

      begin
         null;

         declare
         begin
            if Temp_res_1 /= 0 then
               Ada.Exceptions.Raise_Exception
                 (CUDA.Exception_Registry.Element (Integer (Temp_res_1)));
            end if;
         end;
      end;
   end Set_Valid_Devices;

   procedure Set_Device_Flags (Flags : unsigned) is

      Temp_local_3 : aliased unsigned with
         Address => Flags'Address,
         Import;

   begin
      declare

         Temp_res_1 : Integer :=
           Integer (cuda_runtime_api_h.cudaSetDeviceFlags (Temp_local_3));

      begin
         null;

         declare
         begin
            if Temp_res_1 /= 0 then
               Ada.Exceptions.Raise_Exception
                 (CUDA.Exception_Registry.Element (Integer (Temp_res_1)));
            end if;
         end;
      end;
   end Set_Device_Flags;

   function Get_Device_Flags return unsigned is

      Temp_call_1 : aliased unsigned;
      Temp_ret_2  : aliased unsigned with
         Address => Temp_call_1'Address,
         Import;

   begin
      declare

         Temp_res_1 : Integer :=
           Integer
             (cuda_runtime_api_h.cudaGetDeviceFlags
                (Temp_call_1'Unchecked_Access));

      begin
         null;

         declare
         begin
            if Temp_res_1 /= 0 then
               Ada.Exceptions.Raise_Exception
                 (CUDA.Exception_Registry.Element (Integer (Temp_res_1)));
            end if;
            return Temp_ret_2;
         end;
      end;
   end Get_Device_Flags;

   procedure Stream_Create (P_Stream : System.Address) is

      Temp_local_3 : aliased System.Address with
         Address => P_Stream'Address,
         Import;

   begin
      declare

         Temp_res_1 : Integer :=
           Integer (cuda_runtime_api_h.cudaStreamCreate (Temp_local_3));

      begin
         null;

         declare
         begin
            if Temp_res_1 /= 0 then
               Ada.Exceptions.Raise_Exception
                 (CUDA.Exception_Registry.Element (Integer (Temp_res_1)));
            end if;
         end;
      end;
   end Stream_Create;

   procedure Stream_Create_With_Flags
     (P_Stream : System.Address; Flags : unsigned)
   is

      Temp_local_3 : aliased System.Address with
         Address => P_Stream'Address,
         Import;
      Temp_local_5 : aliased unsigned with
         Address => Flags'Address,
         Import;

   begin
      declare

         Temp_res_1 : Integer :=
           Integer
             (cuda_runtime_api_h.cudaStreamCreateWithFlags
                (Temp_local_3, Temp_local_5));

      begin
         null;

         declare
         begin
            if Temp_res_1 /= 0 then
               Ada.Exceptions.Raise_Exception
                 (CUDA.Exception_Registry.Element (Integer (Temp_res_1)));
            end if;
         end;
      end;
   end Stream_Create_With_Flags;

   procedure Stream_Create_With_Priority
     (P_Stream : System.Address; Flags : unsigned; Priority : int)
   is

      Temp_local_3 : aliased System.Address with
         Address => P_Stream'Address,
         Import;
      Temp_local_5 : aliased unsigned with
         Address => Flags'Address,
         Import;
      Temp_local_7 : aliased int with
         Address => Priority'Address,
         Import;

   begin
      declare

         Temp_res_1 : Integer :=
           Integer
             (cuda_runtime_api_h.cudaStreamCreateWithPriority
                (Temp_local_3, Temp_local_5, Temp_local_7));

      begin
         null;

         declare
         begin
            if Temp_res_1 /= 0 then
               Ada.Exceptions.Raise_Exception
                 (CUDA.Exception_Registry.Element (Integer (Temp_res_1)));
            end if;
         end;
      end;
   end Stream_Create_With_Priority;

   procedure Stream_Get_Priority
     (H_Stream : CUDA.Driver_Types.Stream_T; Priority : out int)
   is

      Temp_local_3 : aliased driver_types_h.cudaStream_t with
         Address => H_Stream'Address,
         Import;
      Temp_call_4 : aliased int with
         Address => Priority'Address,
         Import;

   begin
      declare

         Temp_res_1 : Integer :=
           Integer
             (cuda_runtime_api_h.cudaStreamGetPriority
                (Temp_local_3, Temp_call_4'Unchecked_Access));

      begin
         null;

         declare
         begin
            if Temp_res_1 /= 0 then
               Ada.Exceptions.Raise_Exception
                 (CUDA.Exception_Registry.Element (Integer (Temp_res_1)));
            end if;
         end;
      end;
   end Stream_Get_Priority;

   procedure Stream_Get_Flags
     (H_Stream : CUDA.Driver_Types.Stream_T; Flags : out unsigned)
   is

      Temp_local_3 : aliased driver_types_h.cudaStream_t with
         Address => H_Stream'Address,
         Import;
      Temp_call_4 : aliased unsigned with
         Address => Flags'Address,
         Import;

   begin
      declare

         Temp_res_1 : Integer :=
           Integer
             (cuda_runtime_api_h.cudaStreamGetFlags
                (Temp_local_3, Temp_call_4'Unchecked_Access));

      begin
         null;

         declare
         begin
            if Temp_res_1 /= 0 then
               Ada.Exceptions.Raise_Exception
                 (CUDA.Exception_Registry.Element (Integer (Temp_res_1)));
            end if;
         end;
      end;
   end Stream_Get_Flags;

   procedure Stream_Destroy (Stream : CUDA.Driver_Types.Stream_T) is

      Temp_local_3 : aliased driver_types_h.cudaStream_t with
         Address => Stream'Address,
         Import;

   begin
      declare

         Temp_res_1 : Integer :=
           Integer (cuda_runtime_api_h.cudaStreamDestroy (Temp_local_3));

      begin
         null;

         declare
         begin
            if Temp_res_1 /= 0 then
               Ada.Exceptions.Raise_Exception
                 (CUDA.Exception_Registry.Element (Integer (Temp_res_1)));
            end if;
         end;
      end;
   end Stream_Destroy;

   procedure Stream_Wait_Event
     (Stream : CUDA.Driver_Types.Stream_T; Event : CUDA.Driver_Types.Event_T;
      Flags  : unsigned)
   is

      Temp_local_3 : aliased driver_types_h.cudaStream_t with
         Address => Stream'Address,
         Import;
      Temp_local_5 : aliased driver_types_h.cudaEvent_t with
         Address => Event'Address,
         Import;
      Temp_local_7 : aliased unsigned with
         Address => Flags'Address,
         Import;

   begin
      declare

         Temp_res_1 : Integer :=
           Integer
             (cuda_runtime_api_h.cudaStreamWaitEvent
                (Temp_local_3, Temp_local_5, Temp_local_7));

      begin
         null;

         declare
         begin
            if Temp_res_1 /= 0 then
               Ada.Exceptions.Raise_Exception
                 (CUDA.Exception_Registry.Element (Integer (Temp_res_1)));
            end if;
         end;
      end;
   end Stream_Wait_Event;

   procedure Stream_Callback_T_Gen
     (Arg1 : driver_types_h.cudaStream_t; Arg2 : driver_types_h.cudaError_t;
      Arg3 : System.Address)
   is

      Temp_local_1 : aliased CUDA.Driver_Types.Stream_T with
         Address => Arg1'Address,
         Import;
      Temp_local_3 : aliased CUDA.Driver_Types.Error_T with
         Address => Arg2'Address,
         Import;
      Temp_local_5 : aliased System.Address with
         Address => Arg3'Address,
         Import;

   begin
      declare
      begin
         Temp_1 (Temp_local_1, Temp_local_3, Temp_local_5);

         declare
         begin
            null;
         end;
      end;
   end Stream_Callback_T_Gen;

   procedure Stream_Add_Callback
     (Stream    : CUDA.Driver_Types.Stream_T; Callback : Stream_Callback_T;
      User_Data : System.Address; Flags : unsigned)
   is

      Temp_local_3 : aliased driver_types_h.cudaStream_t with
         Address => Stream'Address,
         Import;
      Temp_local_5 : aliased cudaStreamCallback_t with
         Address => Callback'Address,
         Import;
      Temp_local_7 : aliased System.Address with
         Address => User_Data'Address,
         Import;
      Temp_local_9 : aliased unsigned with
         Address => Flags'Address,
         Import;

   begin
      declare

         Temp_res_1 : Integer :=
           Integer
             (cuda_runtime_api_h.cudaStreamAddCallback
                (Temp_local_3, Temp_local_5, Temp_local_7, Temp_local_9));

      begin
         null;

         declare
         begin
            if Temp_res_1 /= 0 then
               Ada.Exceptions.Raise_Exception
                 (CUDA.Exception_Registry.Element (Integer (Temp_res_1)));
            end if;
         end;
      end;
   end Stream_Add_Callback;

   procedure Stream_Synchronize (Stream : CUDA.Driver_Types.Stream_T) is

      Temp_local_3 : aliased driver_types_h.cudaStream_t with
         Address => Stream'Address,
         Import;

   begin
      declare

         Temp_res_1 : Integer :=
           Integer (cuda_runtime_api_h.cudaStreamSynchronize (Temp_local_3));

      begin
         null;

         declare
         begin
            if Temp_res_1 /= 0 then
               Ada.Exceptions.Raise_Exception
                 (CUDA.Exception_Registry.Element (Integer (Temp_res_1)));
            end if;
         end;
      end;
   end Stream_Synchronize;

   procedure Stream_Query (Stream : CUDA.Driver_Types.Stream_T) is

      Temp_local_3 : aliased driver_types_h.cudaStream_t with
         Address => Stream'Address,
         Import;

   begin
      declare

         Temp_res_1 : Integer :=
           Integer (cuda_runtime_api_h.cudaStreamQuery (Temp_local_3));

      begin
         null;

         declare
         begin
            if Temp_res_1 /= 0 then
               Ada.Exceptions.Raise_Exception
                 (CUDA.Exception_Registry.Element (Integer (Temp_res_1)));
            end if;
         end;
      end;
   end Stream_Query;

   procedure Stream_Attach_Mem_Async
     (Stream : CUDA.Driver_Types.Stream_T; Dev_Ptr : System.Address;
      Length : CUDA.Corecrt.Size_T; Flags : unsigned)
   is

      Temp_local_3 : aliased driver_types_h.cudaStream_t with
         Address => Stream'Address,
         Import;
      Temp_local_5 : aliased System.Address with
         Address => Dev_Ptr'Address,
         Import;
      Temp_local_7 : aliased corecrt_h.size_t with
         Address => Length'Address,
         Import;
      Temp_local_9 : aliased unsigned with
         Address => Flags'Address,
         Import;

   begin
      declare

         Temp_res_1 : Integer :=
           Integer
             (cuda_runtime_api_h.cudaStreamAttachMemAsync
                (Temp_local_3, Temp_local_5, Temp_local_7, Temp_local_9));

      begin
         null;

         declare
         begin
            if Temp_res_1 /= 0 then
               Ada.Exceptions.Raise_Exception
                 (CUDA.Exception_Registry.Element (Integer (Temp_res_1)));
            end if;
         end;
      end;
   end Stream_Attach_Mem_Async;

   procedure Stream_Begin_Capture
     (Stream : CUDA.Driver_Types.Stream_T;
      Mode   : CUDA.Driver_Types.Stream_Capture_Mode)
   is

      Temp_local_3 : aliased driver_types_h.cudaStream_t with
         Address => Stream'Address,
         Import;
      Temp_local_5 : aliased driver_types_h.cudaStreamCaptureMode with
         Address => Mode'Address,
         Import;

   begin
      declare

         Temp_res_1 : Integer :=
           Integer
             (cuda_runtime_api_h.cudaStreamBeginCapture
                (Temp_local_3, Temp_local_5));

      begin
         null;

         declare
         begin
            if Temp_res_1 /= 0 then
               Ada.Exceptions.Raise_Exception
                 (CUDA.Exception_Registry.Element (Integer (Temp_res_1)));
            end if;
         end;
      end;
   end Stream_Begin_Capture;

   procedure Thread_Exchange_Stream_Capture_Mode
     (Mode : out CUDA.Driver_Types.Stream_Capture_Mode)
   is

      Temp_call_2 : aliased driver_types_h.cudaStreamCaptureMode with
         Address => Mode'Address,
         Import;

   begin
      declare

         Temp_res_1 : Integer :=
           Integer
             (cuda_runtime_api_h.cudaThreadExchangeStreamCaptureMode
                (Temp_call_2'Unchecked_Access));

      begin
         null;

         declare
         begin
            if Temp_res_1 /= 0 then
               Ada.Exceptions.Raise_Exception
                 (CUDA.Exception_Registry.Element (Integer (Temp_res_1)));
            end if;
         end;
      end;
   end Thread_Exchange_Stream_Capture_Mode;

   procedure Stream_End_Capture
     (Stream : CUDA.Driver_Types.Stream_T; P_Graph : System.Address)
   is

      Temp_local_3 : aliased driver_types_h.cudaStream_t with
         Address => Stream'Address,
         Import;
      Temp_local_5 : aliased System.Address with
         Address => P_Graph'Address,
         Import;

   begin
      declare

         Temp_res_1 : Integer :=
           Integer
             (cuda_runtime_api_h.cudaStreamEndCapture
                (Temp_local_3, Temp_local_5));

      begin
         null;

         declare
         begin
            if Temp_res_1 /= 0 then
               Ada.Exceptions.Raise_Exception
                 (CUDA.Exception_Registry.Element (Integer (Temp_res_1)));
            end if;
         end;
      end;
   end Stream_End_Capture;

   procedure Stream_Is_Capturing
     (Stream           :     CUDA.Driver_Types.Stream_T;
      P_Capture_Status : out CUDA.Driver_Types.Stream_Capture_Status)
   is

      Temp_local_3 : aliased driver_types_h.cudaStream_t with
         Address => Stream'Address,
         Import;
      Temp_call_4 : aliased driver_types_h.cudaStreamCaptureStatus with
         Address => P_Capture_Status'Address,
         Import;

   begin
      declare

         Temp_res_1 : Integer :=
           Integer
             (cuda_runtime_api_h.cudaStreamIsCapturing
                (Temp_local_3, Temp_call_4'Unchecked_Access));

      begin
         null;

         declare
         begin
            if Temp_res_1 /= 0 then
               Ada.Exceptions.Raise_Exception
                 (CUDA.Exception_Registry.Element (Integer (Temp_res_1)));
            end if;
         end;
      end;
   end Stream_Is_Capturing;

   procedure Stream_Get_Capture_Info
     (Stream           :     CUDA.Driver_Types.Stream_T;
      P_Capture_Status : out CUDA.Driver_Types.Stream_Capture_Status;
      P_Id             : out Extensions.unsigned_long_long)
   is

      Temp_local_3 : aliased driver_types_h.cudaStream_t with
         Address => Stream'Address,
         Import;
      Temp_call_4 : aliased driver_types_h.cudaStreamCaptureStatus with
         Address => P_Capture_Status'Address,
         Import;
      Temp_call_6 : aliased Extensions.unsigned_long_long with
         Address => P_Id'Address,
         Import;

   begin
      declare

         Temp_res_1 : Integer :=
           Integer
             (cuda_runtime_api_h.cudaStreamGetCaptureInfo
                (Temp_local_3, Temp_call_4'Unchecked_Access,
                 Temp_call_6'Unchecked_Access));

      begin
         null;

         declare
         begin
            if Temp_res_1 /= 0 then
               Ada.Exceptions.Raise_Exception
                 (CUDA.Exception_Registry.Element (Integer (Temp_res_1)));
            end if;
         end;
      end;
   end Stream_Get_Capture_Info;

   function Event_Create return CUDA.Driver_Types.Event_T is

      Temp_ret_2  : aliased CUDA.Driver_Types.Event_T;
      Temp_call_1 : aliased System.Address := Temp_ret_2'Address;

   begin
      declare

         Temp_res_1 : Integer :=
           Integer (cuda_runtime_api_h.cudaEventCreate (Temp_call_1));

      begin
         null;

         declare
         begin
            if Temp_res_1 /= 0 then
               Ada.Exceptions.Raise_Exception
                 (CUDA.Exception_Registry.Element (Integer (Temp_res_1)));
            end if;
            return Temp_ret_2;
         end;
      end;
   end Event_Create;

   function Event_Create_With_Flags
     (Flags : unsigned) return CUDA.Driver_Types.Event_T
   is

      Temp_ret_2   : aliased CUDA.Driver_Types.Event_T;
      Temp_call_1  : aliased System.Address := Temp_ret_2'Address;
      Temp_local_4 : aliased unsigned with
         Address => Flags'Address,
         Import;

   begin
      declare

         Temp_res_1 : Integer :=
           Integer
             (cuda_runtime_api_h.cudaEventCreateWithFlags
                (Temp_call_1, Temp_local_4));

      begin
         null;

         declare
         begin
            if Temp_res_1 /= 0 then
               Ada.Exceptions.Raise_Exception
                 (CUDA.Exception_Registry.Element (Integer (Temp_res_1)));
            end if;
            return Temp_ret_2;
         end;
      end;
   end Event_Create_With_Flags;

   procedure Event_Record
     (Event : CUDA.Driver_Types.Event_T; Stream : CUDA.Driver_Types.Stream_T)
   is

      Temp_local_3 : aliased driver_types_h.cudaEvent_t with
         Address => Event'Address,
         Import;
      Temp_local_5 : aliased driver_types_h.cudaStream_t with
         Address => Stream'Address,
         Import;

   begin
      declare

         Temp_res_1 : Integer :=
           Integer
             (cuda_runtime_api_h.cudaEventRecord (Temp_local_3, Temp_local_5));

      begin
         null;

         declare
         begin
            if Temp_res_1 /= 0 then
               Ada.Exceptions.Raise_Exception
                 (CUDA.Exception_Registry.Element (Integer (Temp_res_1)));
            end if;
         end;
      end;
   end Event_Record;

   procedure Event_Query (Event : CUDA.Driver_Types.Event_T) is

      Temp_local_3 : aliased driver_types_h.cudaEvent_t with
         Address => Event'Address,
         Import;

   begin
      declare

         Temp_res_1 : Integer :=
           Integer (cuda_runtime_api_h.cudaEventQuery (Temp_local_3));

      begin
         null;

         declare
         begin
            if Temp_res_1 /= 0 then
               Ada.Exceptions.Raise_Exception
                 (CUDA.Exception_Registry.Element (Integer (Temp_res_1)));
            end if;
         end;
      end;
   end Event_Query;

   procedure Event_Synchronize (Event : CUDA.Driver_Types.Event_T) is

      Temp_local_3 : aliased driver_types_h.cudaEvent_t with
         Address => Event'Address,
         Import;

   begin
      declare

         Temp_res_1 : Integer :=
           Integer (cuda_runtime_api_h.cudaEventSynchronize (Temp_local_3));

      begin
         null;

         declare
         begin
            if Temp_res_1 /= 0 then
               Ada.Exceptions.Raise_Exception
                 (CUDA.Exception_Registry.Element (Integer (Temp_res_1)));
            end if;
         end;
      end;
   end Event_Synchronize;

   procedure Event_Destroy (Event : CUDA.Driver_Types.Event_T) is

      Temp_local_3 : aliased driver_types_h.cudaEvent_t with
         Address => Event'Address,
         Import;

   begin
      declare

         Temp_res_1 : Integer :=
           Integer (cuda_runtime_api_h.cudaEventDestroy (Temp_local_3));

      begin
         null;

         declare
         begin
            if Temp_res_1 /= 0 then
               Ada.Exceptions.Raise_Exception
                 (CUDA.Exception_Registry.Element (Integer (Temp_res_1)));
            end if;
         end;
      end;
   end Event_Destroy;

   procedure Event_Elapsed_Time
     (Ms    : out Float; Start : CUDA.Driver_Types.Event_T;
      C_End :     CUDA.Driver_Types.Event_T)
   is

      Temp_call_2 : aliased Float with
         Address => Ms'Address,
         Import;
      Temp_local_5 : aliased driver_types_h.cudaEvent_t with
         Address => Start'Address,
         Import;
      Temp_local_7 : aliased driver_types_h.cudaEvent_t with
         Address => C_End'Address,
         Import;

   begin
      declare

         Temp_res_1 : Integer :=
           Integer
             (cuda_runtime_api_h.cudaEventElapsedTime
                (Temp_call_2'Unchecked_Access, Temp_local_5, Temp_local_7));

      begin
         null;

         declare
         begin
            if Temp_res_1 /= 0 then
               Ada.Exceptions.Raise_Exception
                 (CUDA.Exception_Registry.Element (Integer (Temp_res_1)));
            end if;
         end;
      end;
   end Event_Elapsed_Time;

   procedure Import_External_Memory
     (Ext_Mem_Out     :     System.Address;
      Mem_Handle_Desc : out CUDA.Driver_Types.External_Memory_Handle_Desc)
   is

      Temp_local_3 : aliased System.Address with
         Address => Ext_Mem_Out'Address,
         Import;
      Temp_call_4 : aliased driver_types_h.cudaExternalMemoryHandleDesc with
         Address => Mem_Handle_Desc'Address,
         Import;

   begin
      declare

         Temp_res_1 : Integer :=
           Integer
             (cuda_runtime_api_h.cudaImportExternalMemory
                (Temp_local_3, Temp_call_4'Unchecked_Access));

      begin
         null;

         declare
         begin
            if Temp_res_1 /= 0 then
               Ada.Exceptions.Raise_Exception
                 (CUDA.Exception_Registry.Element (Integer (Temp_res_1)));
            end if;
         end;
      end;
   end Import_External_Memory;

   procedure External_Memory_Get_Mapped_Buffer
     (Dev_Ptr : System.Address; Ext_Mem : CUDA.Driver_Types.External_Memory_T;
      Buffer_Desc : out CUDA.Driver_Types.External_Memory_Buffer_Desc)
   is

      Temp_local_3 : aliased System.Address with
         Address => Dev_Ptr'Address,
         Import;
      Temp_local_5 : aliased driver_types_h.cudaExternalMemory_t with
         Address => Ext_Mem'Address,
         Import;
      Temp_call_6 : aliased driver_types_h.cudaExternalMemoryBufferDesc with
         Address => Buffer_Desc'Address,
         Import;

   begin
      declare

         Temp_res_1 : Integer :=
           Integer
             (cuda_runtime_api_h.cudaExternalMemoryGetMappedBuffer
                (Temp_local_3, Temp_local_5, Temp_call_6'Unchecked_Access));

      begin
         null;

         declare
         begin
            if Temp_res_1 /= 0 then
               Ada.Exceptions.Raise_Exception
                 (CUDA.Exception_Registry.Element (Integer (Temp_res_1)));
            end if;
         end;
      end;
   end External_Memory_Get_Mapped_Buffer;

   procedure External_Memory_Get_Mapped_Mipmapped_Array
     (Mipmap : System.Address; Ext_Mem : CUDA.Driver_Types.External_Memory_T;
      Mipmap_Desc : out CUDA.Driver_Types.External_Memory_Mipmapped_Array_Desc)
   is

      Temp_local_3 : aliased System.Address with
         Address => Mipmap'Address,
         Import;
      Temp_local_5 : aliased driver_types_h.cudaExternalMemory_t with
         Address => Ext_Mem'Address,
         Import;
      Temp_call_6 : aliased driver_types_h
        .cudaExternalMemoryMipmappedArrayDesc with
         Address => Mipmap_Desc'Address,
         Import;

   begin
      declare

         Temp_res_1 : Integer :=
           Integer
             (cuda_runtime_api_h.cudaExternalMemoryGetMappedMipmappedArray
                (Temp_local_3, Temp_local_5, Temp_call_6'Unchecked_Access));

      begin
         null;

         declare
         begin
            if Temp_res_1 /= 0 then
               Ada.Exceptions.Raise_Exception
                 (CUDA.Exception_Registry.Element (Integer (Temp_res_1)));
            end if;
         end;
      end;
   end External_Memory_Get_Mapped_Mipmapped_Array;

   procedure Destroy_External_Memory
     (Ext_Mem : CUDA.Driver_Types.External_Memory_T)
   is

      Temp_local_3 : aliased driver_types_h.cudaExternalMemory_t with
         Address => Ext_Mem'Address,
         Import;

   begin
      declare

         Temp_res_1 : Integer :=
           Integer
             (cuda_runtime_api_h.cudaDestroyExternalMemory (Temp_local_3));

      begin
         null;

         declare
         begin
            if Temp_res_1 /= 0 then
               Ada.Exceptions.Raise_Exception
                 (CUDA.Exception_Registry.Element (Integer (Temp_res_1)));
            end if;
         end;
      end;
   end Destroy_External_Memory;

   procedure Import_External_Semaphore
     (Ext_Sem_Out     :     System.Address;
      Sem_Handle_Desc : out CUDA.Driver_Types.External_Semaphore_Handle_Desc)
   is

      Temp_local_3 : aliased System.Address with
         Address => Ext_Sem_Out'Address,
         Import;
      Temp_call_4 : aliased driver_types_h.cudaExternalSemaphoreHandleDesc with
         Address => Sem_Handle_Desc'Address,
         Import;

   begin
      declare

         Temp_res_1 : Integer :=
           Integer
             (cuda_runtime_api_h.cudaImportExternalSemaphore
                (Temp_local_3, Temp_call_4'Unchecked_Access));

      begin
         null;

         declare
         begin
            if Temp_res_1 /= 0 then
               Ada.Exceptions.Raise_Exception
                 (CUDA.Exception_Registry.Element (Integer (Temp_res_1)));
            end if;
         end;
      end;
   end Import_External_Semaphore;

   procedure Signal_External_Semaphores_Async
     (Ext_Sem_Array :     System.Address;
      Params_Array  : out CUDA.Driver_Types.External_Semaphore_Signal_Params;
      Num_Ext_Sems  :     unsigned; Stream : CUDA.Driver_Types.Stream_T)
   is

      Temp_local_3 : aliased System.Address with
         Address => Ext_Sem_Array'Address,
         Import;
      Temp_call_4 : aliased driver_types_h
        .cudaExternalSemaphoreSignalParams with
         Address => Params_Array'Address,
         Import;
      Temp_local_7 : aliased unsigned with
         Address => Num_Ext_Sems'Address,
         Import;
      Temp_local_9 : aliased driver_types_h.cudaStream_t with
         Address => Stream'Address,
         Import;

   begin
      declare

         Temp_res_1 : Integer :=
           Integer
             (cuda_runtime_api_h.cudaSignalExternalSemaphoresAsync
                (Temp_local_3, Temp_call_4'Unchecked_Access, Temp_local_7,
                 Temp_local_9));

      begin
         null;

         declare
         begin
            if Temp_res_1 /= 0 then
               Ada.Exceptions.Raise_Exception
                 (CUDA.Exception_Registry.Element (Integer (Temp_res_1)));
            end if;
         end;
      end;
   end Signal_External_Semaphores_Async;

   procedure Wait_External_Semaphores_Async
     (Ext_Sem_Array :     System.Address;
      Params_Array  : out CUDA.Driver_Types.External_Semaphore_Wait_Params;
      Num_Ext_Sems  :     unsigned; Stream : CUDA.Driver_Types.Stream_T)
   is

      Temp_local_3 : aliased System.Address with
         Address => Ext_Sem_Array'Address,
         Import;
      Temp_call_4 : aliased driver_types_h.cudaExternalSemaphoreWaitParams with
         Address => Params_Array'Address,
         Import;
      Temp_local_7 : aliased unsigned with
         Address => Num_Ext_Sems'Address,
         Import;
      Temp_local_9 : aliased driver_types_h.cudaStream_t with
         Address => Stream'Address,
         Import;

   begin
      declare

         Temp_res_1 : Integer :=
           Integer
             (cuda_runtime_api_h.cudaWaitExternalSemaphoresAsync
                (Temp_local_3, Temp_call_4'Unchecked_Access, Temp_local_7,
                 Temp_local_9));

      begin
         null;

         declare
         begin
            if Temp_res_1 /= 0 then
               Ada.Exceptions.Raise_Exception
                 (CUDA.Exception_Registry.Element (Integer (Temp_res_1)));
            end if;
         end;
      end;
   end Wait_External_Semaphores_Async;

   procedure Destroy_External_Semaphore
     (Ext_Sem : CUDA.Driver_Types.External_Semaphore_T)
   is

      Temp_local_3 : aliased driver_types_h.cudaExternalSemaphore_t with
         Address => Ext_Sem'Address,
         Import;

   begin
      declare

         Temp_res_1 : Integer :=
           Integer
             (cuda_runtime_api_h.cudaDestroyExternalSemaphore (Temp_local_3));

      begin
         null;

         declare
         begin
            if Temp_res_1 /= 0 then
               Ada.Exceptions.Raise_Exception
                 (CUDA.Exception_Registry.Element (Integer (Temp_res_1)));
            end if;
         end;
      end;
   end Destroy_External_Semaphore;

   procedure Launch_Kernel
     (Func       : System.Address; Grid_Dim : CUDA.Vector_Types.Dim3;
      Block_Dim  : CUDA.Vector_Types.Dim3; Args : System.Address;
      Shared_Mem : CUDA.Corecrt.Size_T; Stream : CUDA.Driver_Types.Stream_T)
   is

      Temp_local_3 : aliased System.Address with
         Address => Func'Address,
         Import;
      Temp_local_5 : aliased vector_types_h.dim3 with
         Address => Grid_Dim'Address,
         Import;
      Temp_local_7 : aliased vector_types_h.dim3 with
         Address => Block_Dim'Address,
         Import;
      Temp_local_9 : aliased System.Address with
         Address => Args'Address,
         Import;
      Temp_local_11 : aliased corecrt_h.size_t with
         Address => Shared_Mem'Address,
         Import;
      Temp_local_13 : aliased driver_types_h.cudaStream_t with
         Address => Stream'Address,
         Import;

   begin
      declare

         Temp_res_1 : Integer :=
           Integer
             (cuda_runtime_api_h.cudaLaunchKernel
                (Temp_local_3, Temp_local_5, Temp_local_7, Temp_local_9,
                 Temp_local_11, Temp_local_13));

      begin
         null;

         declare
         begin
            if Temp_res_1 /= 0 then
               Ada.Exceptions.Raise_Exception
                 (CUDA.Exception_Registry.Element (Integer (Temp_res_1)));
            end if;
         end;
      end;
   end Launch_Kernel;

   procedure Launch_Cooperative_Kernel
     (Func       : System.Address; Grid_Dim : CUDA.Vector_Types.Dim3;
      Block_Dim  : CUDA.Vector_Types.Dim3; Args : System.Address;
      Shared_Mem : CUDA.Corecrt.Size_T; Stream : CUDA.Driver_Types.Stream_T)
   is

      Temp_local_3 : aliased System.Address with
         Address => Func'Address,
         Import;
      Temp_local_5 : aliased vector_types_h.dim3 with
         Address => Grid_Dim'Address,
         Import;
      Temp_local_7 : aliased vector_types_h.dim3 with
         Address => Block_Dim'Address,
         Import;
      Temp_local_9 : aliased System.Address with
         Address => Args'Address,
         Import;
      Temp_local_11 : aliased corecrt_h.size_t with
         Address => Shared_Mem'Address,
         Import;
      Temp_local_13 : aliased driver_types_h.cudaStream_t with
         Address => Stream'Address,
         Import;

   begin
      declare

         Temp_res_1 : Integer :=
           Integer
             (cuda_runtime_api_h.cudaLaunchCooperativeKernel
                (Temp_local_3, Temp_local_5, Temp_local_7, Temp_local_9,
                 Temp_local_11, Temp_local_13));

      begin
         null;

         declare
         begin
            if Temp_res_1 /= 0 then
               Ada.Exceptions.Raise_Exception
                 (CUDA.Exception_Registry.Element (Integer (Temp_res_1)));
            end if;
         end;
      end;
   end Launch_Cooperative_Kernel;

   procedure Launch_Cooperative_Kernel_Multi_Device
     (Launch_Params_List : out CUDA.Driver_Types.Launch_Params;
      Num_Devices        :     unsigned; Flags : unsigned)
   is

      Temp_call_2 : aliased driver_types_h.cudaLaunchParams with
         Address => Launch_Params_List'Address,
         Import;
      Temp_local_5 : aliased unsigned with
         Address => Num_Devices'Address,
         Import;
      Temp_local_7 : aliased unsigned with
         Address => Flags'Address,
         Import;

   begin
      declare

         Temp_res_1 : Integer :=
           Integer
             (cuda_runtime_api_h.cudaLaunchCooperativeKernelMultiDevice
                (Temp_call_2'Unchecked_Access, Temp_local_5, Temp_local_7));

      begin
         null;

         declare
         begin
            if Temp_res_1 /= 0 then
               Ada.Exceptions.Raise_Exception
                 (CUDA.Exception_Registry.Element (Integer (Temp_res_1)));
            end if;
         end;
      end;
   end Launch_Cooperative_Kernel_Multi_Device;

   procedure Func_Set_Cache_Config
     (Func : System.Address; Cache_Config : CUDA.Driver_Types.Func_Cache)
   is

      Temp_local_3 : aliased System.Address with
         Address => Func'Address,
         Import;
      Temp_local_5 : aliased driver_types_h.cudaFuncCache with
         Address => Cache_Config'Address,
         Import;

   begin
      declare

         Temp_res_1 : Integer :=
           Integer
             (cuda_runtime_api_h.cudaFuncSetCacheConfig
                (Temp_local_3, Temp_local_5));

      begin
         null;

         declare
         begin
            if Temp_res_1 /= 0 then
               Ada.Exceptions.Raise_Exception
                 (CUDA.Exception_Registry.Element (Integer (Temp_res_1)));
            end if;
         end;
      end;
   end Func_Set_Cache_Config;

   procedure Func_Set_Shared_Mem_Config
     (Func : System.Address; Config : CUDA.Driver_Types.Shared_Mem_Config)
   is

      Temp_local_3 : aliased System.Address with
         Address => Func'Address,
         Import;
      Temp_local_5 : aliased driver_types_h.cudaSharedMemConfig with
         Address => Config'Address,
         Import;

   begin
      declare

         Temp_res_1 : Integer :=
           Integer
             (cuda_runtime_api_h.cudaFuncSetSharedMemConfig
                (Temp_local_3, Temp_local_5));

      begin
         null;

         declare
         begin
            if Temp_res_1 /= 0 then
               Ada.Exceptions.Raise_Exception
                 (CUDA.Exception_Registry.Element (Integer (Temp_res_1)));
            end if;
         end;
      end;
   end Func_Set_Shared_Mem_Config;

   function Func_Get_Attributes
     (Func : System.Address) return CUDA.Driver_Types.Func_Attributes
   is

      Temp_call_1 : aliased driver_types_h.cudaFuncAttributes;
      Temp_ret_2  : aliased CUDA.Driver_Types.Func_Attributes with
         Address => Temp_call_1'Address,
         Import;
      Temp_local_4 : aliased System.Address with
         Address => Func'Address,
         Import;

   begin
      declare

         Temp_res_1 : Integer :=
           Integer
             (cuda_runtime_api_h.cudaFuncGetAttributes
                (Temp_call_1'Unchecked_Access, Temp_local_4));

      begin
         null;

         declare
         begin
            if Temp_res_1 /= 0 then
               Ada.Exceptions.Raise_Exception
                 (CUDA.Exception_Registry.Element (Integer (Temp_res_1)));
            end if;
            return Temp_ret_2;
         end;
      end;
   end Func_Get_Attributes;

   procedure Func_Set_Attribute
     (Func  : System.Address; Attr : CUDA.Driver_Types.Func_Attribute;
      Value : int)
   is

      Temp_local_3 : aliased System.Address with
         Address => Func'Address,
         Import;
      Temp_local_5 : aliased driver_types_h.cudaFuncAttribute with
         Address => Attr'Address,
         Import;
      Temp_local_7 : aliased int with
         Address => Value'Address,
         Import;

   begin
      declare

         Temp_res_1 : Integer :=
           Integer
             (cuda_runtime_api_h.cudaFuncSetAttribute
                (Temp_local_3, Temp_local_5, Temp_local_7));

      begin
         null;

         declare
         begin
            if Temp_res_1 /= 0 then
               Ada.Exceptions.Raise_Exception
                 (CUDA.Exception_Registry.Element (Integer (Temp_res_1)));
            end if;
         end;
      end;
   end Func_Set_Attribute;

   procedure Set_Double_For_Device (D : out double) is

      Temp_call_2 : aliased double with
         Address => D'Address,
         Import;

   begin
      declare

         Temp_res_1 : Integer :=
           Integer
             (cuda_runtime_api_h.cudaSetDoubleForDevice
                (Temp_call_2'Unchecked_Access));

      begin
         null;

         declare
         begin
            if Temp_res_1 /= 0 then
               Ada.Exceptions.Raise_Exception
                 (CUDA.Exception_Registry.Element (Integer (Temp_res_1)));
            end if;
         end;
      end;
   end Set_Double_For_Device;

   procedure Set_Double_For_Host (D : out double) is

      Temp_call_2 : aliased double with
         Address => D'Address,
         Import;

   begin
      declare

         Temp_res_1 : Integer :=
           Integer
             (cuda_runtime_api_h.cudaSetDoubleForHost
                (Temp_call_2'Unchecked_Access));

      begin
         null;

         declare
         begin
            if Temp_res_1 /= 0 then
               Ada.Exceptions.Raise_Exception
                 (CUDA.Exception_Registry.Element (Integer (Temp_res_1)));
            end if;
         end;
      end;
   end Set_Double_For_Host;

   procedure Launch_Host_Func
     (Stream    : CUDA.Driver_Types.Stream_T; Fn : CUDA.Driver_Types.Host_Fn_T;
      User_Data : System.Address)
   is

      Temp_local_3 : aliased driver_types_h.cudaStream_t with
         Address => Stream'Address,
         Import;
      Temp_local_5 : aliased driver_types_h.cudaHostFn_t with
         Address => Fn'Address,
         Import;
      Temp_local_7 : aliased System.Address with
         Address => User_Data'Address,
         Import;

   begin
      declare

         Temp_res_1 : Integer :=
           Integer
             (cuda_runtime_api_h.cudaLaunchHostFunc
                (Temp_local_3, Temp_local_5, Temp_local_7));

      begin
         null;

         declare
         begin
            if Temp_res_1 /= 0 then
               Ada.Exceptions.Raise_Exception
                 (CUDA.Exception_Registry.Element (Integer (Temp_res_1)));
            end if;
         end;
      end;
   end Launch_Host_Func;

   procedure Occupancy_Max_Active_Blocks_Per_Multiprocessor
     (Num_Blocks        : out int; Func : System.Address; Block_Size : int;
      Dynamic_SMem_Size :     CUDA.Corecrt.Size_T)
   is

      Temp_call_2 : aliased int with
         Address => Num_Blocks'Address,
         Import;
      Temp_local_5 : aliased System.Address with
         Address => Func'Address,
         Import;
      Temp_local_7 : aliased int with
         Address => Block_Size'Address,
         Import;
      Temp_local_9 : aliased corecrt_h.size_t with
         Address => Dynamic_SMem_Size'Address,
         Import;

   begin
      declare

         Temp_res_1 : Integer :=
           Integer
             (cuda_runtime_api_h.cudaOccupancyMaxActiveBlocksPerMultiprocessor
                (Temp_call_2'Unchecked_Access, Temp_local_5, Temp_local_7,
                 Temp_local_9));

      begin
         null;

         declare
         begin
            if Temp_res_1 /= 0 then
               Ada.Exceptions.Raise_Exception
                 (CUDA.Exception_Registry.Element (Integer (Temp_res_1)));
            end if;
         end;
      end;
   end Occupancy_Max_Active_Blocks_Per_Multiprocessor;

   procedure Occupancy_Max_Active_Blocks_Per_Multiprocessor_With_Flags
     (Num_Blocks        : out int; Func : System.Address; Block_Size : int;
      Dynamic_SMem_Size :     CUDA.Corecrt.Size_T; Flags : unsigned)
   is

      Temp_call_2 : aliased int with
         Address => Num_Blocks'Address,
         Import;
      Temp_local_5 : aliased System.Address with
         Address => Func'Address,
         Import;
      Temp_local_7 : aliased int with
         Address => Block_Size'Address,
         Import;
      Temp_local_9 : aliased corecrt_h.size_t with
         Address => Dynamic_SMem_Size'Address,
         Import;
      Temp_local_11 : aliased unsigned with
         Address => Flags'Address,
         Import;

   begin
      declare

         Temp_res_1 : Integer :=
           Integer
             (cuda_runtime_api_h
                .cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags
                (Temp_call_2'Unchecked_Access, Temp_local_5, Temp_local_7,
                 Temp_local_9, Temp_local_11));

      begin
         null;

         declare
         begin
            if Temp_res_1 /= 0 then
               Ada.Exceptions.Raise_Exception
                 (CUDA.Exception_Registry.Element (Integer (Temp_res_1)));
            end if;
         end;
      end;
   end Occupancy_Max_Active_Blocks_Per_Multiprocessor_With_Flags;

   procedure Malloc_Managed
     (Dev_Ptr : System.Address; Size : CUDA.Corecrt.Size_T; Flags : unsigned)
   is

      Temp_local_3 : aliased System.Address with
         Address => Dev_Ptr'Address,
         Import;
      Temp_local_5 : aliased corecrt_h.size_t with
         Address => Size'Address,
         Import;
      Temp_local_7 : aliased unsigned with
         Address => Flags'Address,
         Import;

   begin
      declare

         Temp_res_1 : Integer :=
           Integer
             (cuda_runtime_api_h.cudaMallocManaged
                (Temp_local_3, Temp_local_5, Temp_local_7));

      begin
         null;

         declare
         begin
            if Temp_res_1 /= 0 then
               Ada.Exceptions.Raise_Exception
                 (CUDA.Exception_Registry.Element (Integer (Temp_res_1)));
            end if;
         end;
      end;
   end Malloc_Managed;

   procedure Malloc (Dev_Ptr : System.Address; Size : CUDA.Corecrt.Size_T) is

      Temp_local_3 : aliased System.Address with
         Address => Dev_Ptr'Address,
         Import;
      Temp_local_5 : aliased corecrt_h.size_t with
         Address => Size'Address,
         Import;

   begin
      declare

         Temp_res_1 : Integer :=
           Integer (cuda_runtime_api_h.cudaMalloc (Temp_local_3, Temp_local_5));

      begin
         null;

         declare
         begin
            if Temp_res_1 /= 0 then
               Ada.Exceptions.Raise_Exception
                 (CUDA.Exception_Registry.Element (Integer (Temp_res_1)));
            end if;
         end;
      end;
   end Malloc;

   procedure Malloc_Host (Ptr : System.Address; Size : CUDA.Corecrt.Size_T) is

      Temp_local_3 : aliased System.Address with
         Address => Ptr'Address,
         Import;
      Temp_local_5 : aliased corecrt_h.size_t with
         Address => Size'Address,
         Import;

   begin
      declare

         Temp_res_1 : Integer :=
           Integer
             (cuda_runtime_api_h.cudaMallocHost (Temp_local_3, Temp_local_5));

      begin
         null;

         declare
         begin
            if Temp_res_1 /= 0 then
               Ada.Exceptions.Raise_Exception
                 (CUDA.Exception_Registry.Element (Integer (Temp_res_1)));
            end if;
         end;
      end;
   end Malloc_Host;

   procedure Malloc_Pitch
     (Dev_Ptr : System.Address; Pitch : out CUDA.Corecrt.Size_T;
      Width   : CUDA.Corecrt.Size_T; Height : CUDA.Corecrt.Size_T)
   is

      Temp_local_3 : aliased System.Address with
         Address => Dev_Ptr'Address,
         Import;
      Temp_call_4 : aliased corecrt_h.size_t with
         Address => Pitch'Address,
         Import;
      Temp_local_7 : aliased corecrt_h.size_t with
         Address => Width'Address,
         Import;
      Temp_local_9 : aliased corecrt_h.size_t with
         Address => Height'Address,
         Import;

   begin
      declare

         Temp_res_1 : Integer :=
           Integer
             (cuda_runtime_api_h.cudaMallocPitch
                (Temp_local_3, Temp_call_4'Unchecked_Access, Temp_local_7,
                 Temp_local_9));

      begin
         null;

         declare
         begin
            if Temp_res_1 /= 0 then
               Ada.Exceptions.Raise_Exception
                 (CUDA.Exception_Registry.Element (Integer (Temp_res_1)));
            end if;
         end;
      end;
   end Malloc_Pitch;

   procedure Malloc_Array
     (C_Array :     System.Address;
      Desc    : out CUDA.Driver_Types.Channel_Format_Desc;
      Width   :     CUDA.Corecrt.Size_T; Height : CUDA.Corecrt.Size_T;
      Flags   :     unsigned)
   is

      Temp_local_3 : aliased System.Address with
         Address => C_Array'Address,
         Import;
      Temp_call_4 : aliased driver_types_h.cudaChannelFormatDesc with
         Address => Desc'Address,
         Import;
      Temp_local_7 : aliased corecrt_h.size_t with
         Address => Width'Address,
         Import;
      Temp_local_9 : aliased corecrt_h.size_t with
         Address => Height'Address,
         Import;
      Temp_local_11 : aliased unsigned with
         Address => Flags'Address,
         Import;

   begin
      declare

         Temp_res_1 : Integer :=
           Integer
             (cuda_runtime_api_h.cudaMallocArray
                (Temp_local_3, Temp_call_4'Unchecked_Access, Temp_local_7,
                 Temp_local_9, Temp_local_11));

      begin
         null;

         declare
         begin
            if Temp_res_1 /= 0 then
               Ada.Exceptions.Raise_Exception
                 (CUDA.Exception_Registry.Element (Integer (Temp_res_1)));
            end if;
         end;
      end;
   end Malloc_Array;

   procedure Free (Dev_Ptr : System.Address) is

      Temp_local_3 : aliased System.Address with
         Address => Dev_Ptr'Address,
         Import;

   begin
      declare

         Temp_res_1 : Integer :=
           Integer (cuda_runtime_api_h.cudaFree (Temp_local_3));

      begin
         null;

         declare
         begin
            if Temp_res_1 /= 0 then
               Ada.Exceptions.Raise_Exception
                 (CUDA.Exception_Registry.Element (Integer (Temp_res_1)));
            end if;
         end;
      end;
   end Free;

   procedure Free_Host (Ptr : System.Address) is

      Temp_local_3 : aliased System.Address with
         Address => Ptr'Address,
         Import;

   begin
      declare

         Temp_res_1 : Integer :=
           Integer (cuda_runtime_api_h.cudaFreeHost (Temp_local_3));

      begin
         null;

         declare
         begin
            if Temp_res_1 /= 0 then
               Ada.Exceptions.Raise_Exception
                 (CUDA.Exception_Registry.Element (Integer (Temp_res_1)));
            end if;
         end;
      end;
   end Free_Host;

   procedure Free_Array (C_Array : CUDA.Driver_Types.CUDA_Array_t) is

      Temp_local_3 : aliased driver_types_h.cudaArray_t with
         Address => C_Array'Address,
         Import;

   begin
      declare

         Temp_res_1 : Integer :=
           Integer (cuda_runtime_api_h.cudaFreeArray (Temp_local_3));

      begin
         null;

         declare
         begin
            if Temp_res_1 /= 0 then
               Ada.Exceptions.Raise_Exception
                 (CUDA.Exception_Registry.Element (Integer (Temp_res_1)));
            end if;
         end;
      end;
   end Free_Array;

   procedure Free_Mipmapped_Array
     (Mipmapped_Array : CUDA.Driver_Types.Mipmapped_Array_T)
   is

      Temp_local_3 : aliased driver_types_h.cudaMipmappedArray_t with
         Address => Mipmapped_Array'Address,
         Import;

   begin
      declare

         Temp_res_1 : Integer :=
           Integer (cuda_runtime_api_h.cudaFreeMipmappedArray (Temp_local_3));

      begin
         null;

         declare
         begin
            if Temp_res_1 /= 0 then
               Ada.Exceptions.Raise_Exception
                 (CUDA.Exception_Registry.Element (Integer (Temp_res_1)));
            end if;
         end;
      end;
   end Free_Mipmapped_Array;

   procedure Host_Alloc
     (P_Host : System.Address; Size : CUDA.Corecrt.Size_T; Flags : unsigned)
   is

      Temp_local_3 : aliased System.Address with
         Address => P_Host'Address,
         Import;
      Temp_local_5 : aliased corecrt_h.size_t with
         Address => Size'Address,
         Import;
      Temp_local_7 : aliased unsigned with
         Address => Flags'Address,
         Import;

   begin
      declare

         Temp_res_1 : Integer :=
           Integer
             (cuda_runtime_api_h.cudaHostAlloc
                (Temp_local_3, Temp_local_5, Temp_local_7));

      begin
         null;

         declare
         begin
            if Temp_res_1 /= 0 then
               Ada.Exceptions.Raise_Exception
                 (CUDA.Exception_Registry.Element (Integer (Temp_res_1)));
            end if;
         end;
      end;
   end Host_Alloc;

   procedure Host_Register
     (Ptr : System.Address; Size : CUDA.Corecrt.Size_T; Flags : unsigned)
   is

      Temp_local_3 : aliased System.Address with
         Address => Ptr'Address,
         Import;
      Temp_local_5 : aliased corecrt_h.size_t with
         Address => Size'Address,
         Import;
      Temp_local_7 : aliased unsigned with
         Address => Flags'Address,
         Import;

   begin
      declare

         Temp_res_1 : Integer :=
           Integer
             (cuda_runtime_api_h.cudaHostRegister
                (Temp_local_3, Temp_local_5, Temp_local_7));

      begin
         null;

         declare
         begin
            if Temp_res_1 /= 0 then
               Ada.Exceptions.Raise_Exception
                 (CUDA.Exception_Registry.Element (Integer (Temp_res_1)));
            end if;
         end;
      end;
   end Host_Register;

   procedure Host_Unregister (Ptr : System.Address) is

      Temp_local_3 : aliased System.Address with
         Address => Ptr'Address,
         Import;

   begin
      declare

         Temp_res_1 : Integer :=
           Integer (cuda_runtime_api_h.cudaHostUnregister (Temp_local_3));

      begin
         null;

         declare
         begin
            if Temp_res_1 /= 0 then
               Ada.Exceptions.Raise_Exception
                 (CUDA.Exception_Registry.Element (Integer (Temp_res_1)));
            end if;
         end;
      end;
   end Host_Unregister;

   procedure Host_Get_Device_Pointer
     (P_Device : System.Address; P_Host : System.Address; Flags : unsigned)
   is

      Temp_local_3 : aliased System.Address with
         Address => P_Device'Address,
         Import;
      Temp_local_5 : aliased System.Address with
         Address => P_Host'Address,
         Import;
      Temp_local_7 : aliased unsigned with
         Address => Flags'Address,
         Import;

   begin
      declare

         Temp_res_1 : Integer :=
           Integer
             (cuda_runtime_api_h.cudaHostGetDevicePointer
                (Temp_local_3, Temp_local_5, Temp_local_7));

      begin
         null;

         declare
         begin
            if Temp_res_1 /= 0 then
               Ada.Exceptions.Raise_Exception
                 (CUDA.Exception_Registry.Element (Integer (Temp_res_1)));
            end if;
         end;
      end;
   end Host_Get_Device_Pointer;

   function Host_Get_Flags (P_Host : System.Address) return unsigned is

      Temp_call_1 : aliased unsigned;
      Temp_ret_2  : aliased unsigned with
         Address => Temp_call_1'Address,
         Import;
      Temp_local_4 : aliased System.Address with
         Address => P_Host'Address,
         Import;

   begin
      declare

         Temp_res_1 : Integer :=
           Integer
             (cuda_runtime_api_h.cudaHostGetFlags
                (Temp_call_1'Unchecked_Access, Temp_local_4));

      begin
         null;

         declare
         begin
            if Temp_res_1 /= 0 then
               Ada.Exceptions.Raise_Exception
                 (CUDA.Exception_Registry.Element (Integer (Temp_res_1)));
            end if;
            return Temp_ret_2;
         end;
      end;
   end Host_Get_Flags;

   procedure Malloc3_D
     (Pitched_Dev_Ptr : out CUDA.Driver_Types.Pitched_Ptr;
      Extent          :     CUDA.Driver_Types.Extent_T)
   is

      Temp_call_2 : aliased driver_types_h.cudaPitchedPtr with
         Address => Pitched_Dev_Ptr'Address,
         Import;
      Temp_local_5 : aliased driver_types_h.cudaExtent with
         Address => Extent'Address,
         Import;

   begin
      declare

         Temp_res_1 : Integer :=
           Integer
             (cuda_runtime_api_h.cudaMalloc3D
                (Temp_call_2'Unchecked_Access, Temp_local_5));

      begin
         null;

         declare
         begin
            if Temp_res_1 /= 0 then
               Ada.Exceptions.Raise_Exception
                 (CUDA.Exception_Registry.Element (Integer (Temp_res_1)));
            end if;
         end;
      end;
   end Malloc3_D;

   procedure Malloc3_DArray
     (C_Array :     System.Address;
      Desc    : out CUDA.Driver_Types.Channel_Format_Desc;
      Extent  :     CUDA.Driver_Types.Extent_T; Flags : unsigned)
   is

      Temp_local_3 : aliased System.Address with
         Address => C_Array'Address,
         Import;
      Temp_call_4 : aliased driver_types_h.cudaChannelFormatDesc with
         Address => Desc'Address,
         Import;
      Temp_local_7 : aliased driver_types_h.cudaExtent with
         Address => Extent'Address,
         Import;
      Temp_local_9 : aliased unsigned with
         Address => Flags'Address,
         Import;

   begin
      declare

         Temp_res_1 : Integer :=
           Integer
             (cuda_runtime_api_h.cudaMalloc3DArray
                (Temp_local_3, Temp_call_4'Unchecked_Access, Temp_local_7,
                 Temp_local_9));

      begin
         null;

         declare
         begin
            if Temp_res_1 /= 0 then
               Ada.Exceptions.Raise_Exception
                 (CUDA.Exception_Registry.Element (Integer (Temp_res_1)));
            end if;
         end;
      end;
   end Malloc3_DArray;

   procedure Malloc_Mipmapped_Array
     (Mipmapped_Array :     System.Address;
      Desc            : out CUDA.Driver_Types.Channel_Format_Desc;
      Extent          :     CUDA.Driver_Types.Extent_T; Num_Levels : unsigned;
      Flags           :     unsigned)
   is

      Temp_local_3 : aliased System.Address with
         Address => Mipmapped_Array'Address,
         Import;
      Temp_call_4 : aliased driver_types_h.cudaChannelFormatDesc with
         Address => Desc'Address,
         Import;
      Temp_local_7 : aliased driver_types_h.cudaExtent with
         Address => Extent'Address,
         Import;
      Temp_local_9 : aliased unsigned with
         Address => Num_Levels'Address,
         Import;
      Temp_local_11 : aliased unsigned with
         Address => Flags'Address,
         Import;

   begin
      declare

         Temp_res_1 : Integer :=
           Integer
             (cuda_runtime_api_h.cudaMallocMipmappedArray
                (Temp_local_3, Temp_call_4'Unchecked_Access, Temp_local_7,
                 Temp_local_9, Temp_local_11));

      begin
         null;

         declare
         begin
            if Temp_res_1 /= 0 then
               Ada.Exceptions.Raise_Exception
                 (CUDA.Exception_Registry.Element (Integer (Temp_res_1)));
            end if;
         end;
      end;
   end Malloc_Mipmapped_Array;

   procedure Get_Mipmapped_Array_Level
     (Level_Array     : System.Address;
      Mipmapped_Array : CUDA.Driver_Types.Mipmapped_Array_Const_T;
      Level           : unsigned)
   is

      Temp_local_3 : aliased System.Address with
         Address => Level_Array'Address,
         Import;
      Temp_local_5 : aliased driver_types_h.cudaMipmappedArray_const_t with
         Address => Mipmapped_Array'Address,
         Import;
      Temp_local_7 : aliased unsigned with
         Address => Level'Address,
         Import;

   begin
      declare

         Temp_res_1 : Integer :=
           Integer
             (cuda_runtime_api_h.cudaGetMipmappedArrayLevel
                (Temp_local_3, Temp_local_5, Temp_local_7));

      begin
         null;

         declare
         begin
            if Temp_res_1 /= 0 then
               Ada.Exceptions.Raise_Exception
                 (CUDA.Exception_Registry.Element (Integer (Temp_res_1)));
            end if;
         end;
      end;
   end Get_Mipmapped_Array_Level;

   procedure Memcpy3_D (P : out CUDA.Driver_Types.Memcpy3_DParms) is

      Temp_call_2 : aliased driver_types_h.cudaMemcpy3DParms with
         Address => P'Address,
         Import;

   begin
      declare

         Temp_res_1 : Integer :=
           Integer
             (cuda_runtime_api_h.cudaMemcpy3D (Temp_call_2'Unchecked_Access));

      begin
         null;

         declare
         begin
            if Temp_res_1 /= 0 then
               Ada.Exceptions.Raise_Exception
                 (CUDA.Exception_Registry.Element (Integer (Temp_res_1)));
            end if;
         end;
      end;
   end Memcpy3_D;

   procedure Memcpy3_DPeer (P : out CUDA.Driver_Types.Memcpy3_DPeer_Parms) is

      Temp_call_2 : aliased driver_types_h.cudaMemcpy3DPeerParms with
         Address => P'Address,
         Import;

   begin
      declare

         Temp_res_1 : Integer :=
           Integer
             (cuda_runtime_api_h.cudaMemcpy3DPeer
                (Temp_call_2'Unchecked_Access));

      begin
         null;

         declare
         begin
            if Temp_res_1 /= 0 then
               Ada.Exceptions.Raise_Exception
                 (CUDA.Exception_Registry.Element (Integer (Temp_res_1)));
            end if;
         end;
      end;
   end Memcpy3_DPeer;

   procedure Memcpy3_DAsync
     (P      : out CUDA.Driver_Types.Memcpy3_DParms;
      Stream :     CUDA.Driver_Types.Stream_T)
   is

      Temp_call_2 : aliased driver_types_h.cudaMemcpy3DParms with
         Address => P'Address,
         Import;
      Temp_local_5 : aliased driver_types_h.cudaStream_t with
         Address => Stream'Address,
         Import;

   begin
      declare

         Temp_res_1 : Integer :=
           Integer
             (cuda_runtime_api_h.cudaMemcpy3DAsync
                (Temp_call_2'Unchecked_Access, Temp_local_5));

      begin
         null;

         declare
         begin
            if Temp_res_1 /= 0 then
               Ada.Exceptions.Raise_Exception
                 (CUDA.Exception_Registry.Element (Integer (Temp_res_1)));
            end if;
         end;
      end;
   end Memcpy3_DAsync;

   procedure Memcpy3_DPeer_Async
     (P      : out CUDA.Driver_Types.Memcpy3_DPeer_Parms;
      Stream :     CUDA.Driver_Types.Stream_T)
   is

      Temp_call_2 : aliased driver_types_h.cudaMemcpy3DPeerParms with
         Address => P'Address,
         Import;
      Temp_local_5 : aliased driver_types_h.cudaStream_t with
         Address => Stream'Address,
         Import;

   begin
      declare

         Temp_res_1 : Integer :=
           Integer
             (cuda_runtime_api_h.cudaMemcpy3DPeerAsync
                (Temp_call_2'Unchecked_Access, Temp_local_5));

      begin
         null;

         declare
         begin
            if Temp_res_1 /= 0 then
               Ada.Exceptions.Raise_Exception
                 (CUDA.Exception_Registry.Element (Integer (Temp_res_1)));
            end if;
         end;
      end;
   end Memcpy3_DPeer_Async;

   function Mem_Get_Info
     (Total : out CUDA.Corecrt.Size_T) return CUDA.Corecrt.Size_T
   is

      Temp_call_1 : aliased corecrt_h.size_t;
      Temp_ret_2  : aliased CUDA.Corecrt.Size_T with
         Address => Temp_call_1'Address,
         Import;
      Temp_call_3 : aliased corecrt_h.size_t with
         Address => Total'Address,
         Import;

   begin
      declare

         Temp_res_1 : Integer :=
           Integer
             (cuda_runtime_api_h.cudaMemGetInfo
                (Temp_call_1'Unchecked_Access, Temp_call_3'Unchecked_Access));

      begin
         null;

         declare
         begin
            if Temp_res_1 /= 0 then
               Ada.Exceptions.Raise_Exception
                 (CUDA.Exception_Registry.Element (Integer (Temp_res_1)));
            end if;
            return Temp_ret_2;
         end;
      end;
   end Mem_Get_Info;

   function CUDA_ArrayGetInfo
     (Extent  : out CUDA.Driver_Types.Extent_T; Flags : out unsigned;
      C_Array :     CUDA.Driver_Types.CUDA_Array_t)
      return CUDA.Driver_Types.Channel_Format_Desc
   is

      Temp_call_1 : aliased driver_types_h.cudaChannelFormatDesc;
      Temp_ret_2  : aliased CUDA.Driver_Types.Channel_Format_Desc with
         Address => Temp_call_1'Address,
         Import;
      Temp_call_3 : aliased driver_types_h.cudaExtent with
         Address => Extent'Address,
         Import;
      Temp_call_5 : aliased unsigned with
         Address => Flags'Address,
         Import;
      Temp_local_8 : aliased driver_types_h.cudaArray_t with
         Address => C_Array'Address,
         Import;

   begin
      declare

         Temp_res_1 : Integer :=
           Integer
             (cuda_runtime_api_h.cudaArrayGetInfo
                (Temp_call_1'Unchecked_Access, Temp_call_3'Unchecked_Access,
                 Temp_call_5'Unchecked_Access, Temp_local_8));

      begin
         null;

         declare
         begin
            if Temp_res_1 /= 0 then
               Ada.Exceptions.Raise_Exception
                 (CUDA.Exception_Registry.Element (Integer (Temp_res_1)));
            end if;
            return Temp_ret_2;
         end;
      end;
   end CUDA_ArrayGetInfo;

   procedure Memcpy
     (Dst  : System.Address; Src : System.Address; Count : CUDA.Corecrt.Size_T;
      Kind : CUDA.Driver_Types.Memcpy_Kind)
   is

      Temp_local_3 : aliased System.Address with
         Address => Dst'Address,
         Import;
      Temp_local_5 : aliased System.Address with
         Address => Src'Address,
         Import;
      Temp_local_7 : aliased corecrt_h.size_t with
         Address => Count'Address,
         Import;
      Temp_local_9 : aliased driver_types_h.cudaMemcpyKind with
         Address => Kind'Address,
         Import;

   begin
      declare

         Temp_res_1 : Integer :=
           Integer
             (cuda_runtime_api_h.cudaMemcpy
                (Temp_local_3, Temp_local_5, Temp_local_7, Temp_local_9));

      begin
         null;

         declare
         begin
            if Temp_res_1 /= 0 then
               Ada.Exceptions.Raise_Exception
                 (CUDA.Exception_Registry.Element (Integer (Temp_res_1)));
            end if;
         end;
      end;
   end Memcpy;

   procedure Memcpy_Peer
     (Dst        : System.Address; Dst_Device : int; Src : System.Address;
      Src_Device : int; Count : CUDA.Corecrt.Size_T)
   is

      Temp_local_3 : aliased System.Address with
         Address => Dst'Address,
         Import;
      Temp_local_5 : aliased int with
         Address => Dst_Device'Address,
         Import;
      Temp_local_7 : aliased System.Address with
         Address => Src'Address,
         Import;
      Temp_local_9 : aliased int with
         Address => Src_Device'Address,
         Import;
      Temp_local_11 : aliased corecrt_h.size_t with
         Address => Count'Address,
         Import;

   begin
      declare

         Temp_res_1 : Integer :=
           Integer
             (cuda_runtime_api_h.cudaMemcpyPeer
                (Temp_local_3, Temp_local_5, Temp_local_7, Temp_local_9,
                 Temp_local_11));

      begin
         null;

         declare
         begin
            if Temp_res_1 /= 0 then
               Ada.Exceptions.Raise_Exception
                 (CUDA.Exception_Registry.Element (Integer (Temp_res_1)));
            end if;
         end;
      end;
   end Memcpy_Peer;

   procedure Memcpy2_D
     (Dst : System.Address; Dpitch : CUDA.Corecrt.Size_T; Src : System.Address;
      Spitch : CUDA.Corecrt.Size_T; Width : CUDA.Corecrt.Size_T;
      Height : CUDA.Corecrt.Size_T; Kind : CUDA.Driver_Types.Memcpy_Kind)
   is

      Temp_local_3 : aliased System.Address with
         Address => Dst'Address,
         Import;
      Temp_local_5 : aliased corecrt_h.size_t with
         Address => Dpitch'Address,
         Import;
      Temp_local_7 : aliased System.Address with
         Address => Src'Address,
         Import;
      Temp_local_9 : aliased corecrt_h.size_t with
         Address => Spitch'Address,
         Import;
      Temp_local_11 : aliased corecrt_h.size_t with
         Address => Width'Address,
         Import;
      Temp_local_13 : aliased corecrt_h.size_t with
         Address => Height'Address,
         Import;
      Temp_local_15 : aliased driver_types_h.cudaMemcpyKind with
         Address => Kind'Address,
         Import;

   begin
      declare

         Temp_res_1 : Integer :=
           Integer
             (cuda_runtime_api_h.cudaMemcpy2D
                (Temp_local_3, Temp_local_5, Temp_local_7, Temp_local_9,
                 Temp_local_11, Temp_local_13, Temp_local_15));

      begin
         null;

         declare
         begin
            if Temp_res_1 /= 0 then
               Ada.Exceptions.Raise_Exception
                 (CUDA.Exception_Registry.Element (Integer (Temp_res_1)));
            end if;
         end;
      end;
   end Memcpy2_D;

   procedure Memcpy2_DTo_Array
     (Dst      : CUDA.Driver_Types.CUDA_Array_t; W_Offset : CUDA.Corecrt.Size_T;
      H_Offset : CUDA.Corecrt.Size_T; Src : System.Address;
      Spitch   : CUDA.Corecrt.Size_T; Width : CUDA.Corecrt.Size_T;
      Height   : CUDA.Corecrt.Size_T; Kind : CUDA.Driver_Types.Memcpy_Kind)
   is

      Temp_local_3 : aliased driver_types_h.cudaArray_t with
         Address => Dst'Address,
         Import;
      Temp_local_5 : aliased corecrt_h.size_t with
         Address => W_Offset'Address,
         Import;
      Temp_local_7 : aliased corecrt_h.size_t with
         Address => H_Offset'Address,
         Import;
      Temp_local_9 : aliased System.Address with
         Address => Src'Address,
         Import;
      Temp_local_11 : aliased corecrt_h.size_t with
         Address => Spitch'Address,
         Import;
      Temp_local_13 : aliased corecrt_h.size_t with
         Address => Width'Address,
         Import;
      Temp_local_15 : aliased corecrt_h.size_t with
         Address => Height'Address,
         Import;
      Temp_local_17 : aliased driver_types_h.cudaMemcpyKind with
         Address => Kind'Address,
         Import;

   begin
      declare

         Temp_res_1 : Integer :=
           Integer
             (cuda_runtime_api_h.cudaMemcpy2DToArray
                (Temp_local_3, Temp_local_5, Temp_local_7, Temp_local_9,
                 Temp_local_11, Temp_local_13, Temp_local_15, Temp_local_17));

      begin
         null;

         declare
         begin
            if Temp_res_1 /= 0 then
               Ada.Exceptions.Raise_Exception
                 (CUDA.Exception_Registry.Element (Integer (Temp_res_1)));
            end if;
         end;
      end;
   end Memcpy2_DTo_Array;

   procedure Memcpy2_DFrom_Array
     (Dst      : System.Address; Dpitch : CUDA.Corecrt.Size_T;
      Src      : CUDA.Driver_Types.CUDA_Array_const_t;
      W_Offset : CUDA.Corecrt.Size_T; H_Offset : CUDA.Corecrt.Size_T;
      Width    : CUDA.Corecrt.Size_T; Height : CUDA.Corecrt.Size_T;
      Kind     : CUDA.Driver_Types.Memcpy_Kind)
   is

      Temp_local_3 : aliased System.Address with
         Address => Dst'Address,
         Import;
      Temp_local_5 : aliased corecrt_h.size_t with
         Address => Dpitch'Address,
         Import;
      Temp_local_7 : aliased driver_types_h.cudaArray_const_t with
         Address => Src'Address,
         Import;
      Temp_local_9 : aliased corecrt_h.size_t with
         Address => W_Offset'Address,
         Import;
      Temp_local_11 : aliased corecrt_h.size_t with
         Address => H_Offset'Address,
         Import;
      Temp_local_13 : aliased corecrt_h.size_t with
         Address => Width'Address,
         Import;
      Temp_local_15 : aliased corecrt_h.size_t with
         Address => Height'Address,
         Import;
      Temp_local_17 : aliased driver_types_h.cudaMemcpyKind with
         Address => Kind'Address,
         Import;

   begin
      declare

         Temp_res_1 : Integer :=
           Integer
             (cuda_runtime_api_h.cudaMemcpy2DFromArray
                (Temp_local_3, Temp_local_5, Temp_local_7, Temp_local_9,
                 Temp_local_11, Temp_local_13, Temp_local_15, Temp_local_17));

      begin
         null;

         declare
         begin
            if Temp_res_1 /= 0 then
               Ada.Exceptions.Raise_Exception
                 (CUDA.Exception_Registry.Element (Integer (Temp_res_1)));
            end if;
         end;
      end;
   end Memcpy2_DFrom_Array;

   procedure Memcpy2_DArray_To_Array
     (Dst : CUDA.Driver_Types.CUDA_Array_t; W_Offset_Dst : CUDA.Corecrt.Size_T;
      H_Offset_Dst : CUDA.Corecrt.Size_T;
      Src          : CUDA.Driver_Types.CUDA_Array_const_t;
      W_Offset_Src : CUDA.Corecrt.Size_T; H_Offset_Src : CUDA.Corecrt.Size_T;
      Width        : CUDA.Corecrt.Size_T; Height : CUDA.Corecrt.Size_T;
      Kind         : CUDA.Driver_Types.Memcpy_Kind)
   is

      Temp_local_3 : aliased driver_types_h.cudaArray_t with
         Address => Dst'Address,
         Import;
      Temp_local_5 : aliased corecrt_h.size_t with
         Address => W_Offset_Dst'Address,
         Import;
      Temp_local_7 : aliased corecrt_h.size_t with
         Address => H_Offset_Dst'Address,
         Import;
      Temp_local_9 : aliased driver_types_h.cudaArray_const_t with
         Address => Src'Address,
         Import;
      Temp_local_11 : aliased corecrt_h.size_t with
         Address => W_Offset_Src'Address,
         Import;
      Temp_local_13 : aliased corecrt_h.size_t with
         Address => H_Offset_Src'Address,
         Import;
      Temp_local_15 : aliased corecrt_h.size_t with
         Address => Width'Address,
         Import;
      Temp_local_17 : aliased corecrt_h.size_t with
         Address => Height'Address,
         Import;
      Temp_local_19 : aliased driver_types_h.cudaMemcpyKind with
         Address => Kind'Address,
         Import;

   begin
      declare

         Temp_res_1 : Integer :=
           Integer
             (cuda_runtime_api_h.cudaMemcpy2DArrayToArray
                (Temp_local_3, Temp_local_5, Temp_local_7, Temp_local_9,
                 Temp_local_11, Temp_local_13, Temp_local_15, Temp_local_17,
                 Temp_local_19));

      begin
         null;

         declare
         begin
            if Temp_res_1 /= 0 then
               Ada.Exceptions.Raise_Exception
                 (CUDA.Exception_Registry.Element (Integer (Temp_res_1)));
            end if;
         end;
      end;
   end Memcpy2_DArray_To_Array;

   procedure Memcpy_To_Symbol
     (Symbol : System.Address; Src : System.Address;
      Count  : CUDA.Corecrt.Size_T; Offset : CUDA.Corecrt.Size_T;
      Kind   : CUDA.Driver_Types.Memcpy_Kind)
   is

      Temp_local_3 : aliased System.Address with
         Address => Symbol'Address,
         Import;
      Temp_local_5 : aliased System.Address with
         Address => Src'Address,
         Import;
      Temp_local_7 : aliased corecrt_h.size_t with
         Address => Count'Address,
         Import;
      Temp_local_9 : aliased corecrt_h.size_t with
         Address => Offset'Address,
         Import;
      Temp_local_11 : aliased driver_types_h.cudaMemcpyKind with
         Address => Kind'Address,
         Import;

   begin
      declare

         Temp_res_1 : Integer :=
           Integer
             (cuda_runtime_api_h.cudaMemcpyToSymbol
                (Temp_local_3, Temp_local_5, Temp_local_7, Temp_local_9,
                 Temp_local_11));

      begin
         null;

         declare
         begin
            if Temp_res_1 /= 0 then
               Ada.Exceptions.Raise_Exception
                 (CUDA.Exception_Registry.Element (Integer (Temp_res_1)));
            end if;
         end;
      end;
   end Memcpy_To_Symbol;

   procedure Memcpy_From_Symbol
     (Dst   : System.Address; Symbol : System.Address;
      Count : CUDA.Corecrt.Size_T; Offset : CUDA.Corecrt.Size_T;
      Kind  : CUDA.Driver_Types.Memcpy_Kind)
   is

      Temp_local_3 : aliased System.Address with
         Address => Dst'Address,
         Import;
      Temp_local_5 : aliased System.Address with
         Address => Symbol'Address,
         Import;
      Temp_local_7 : aliased corecrt_h.size_t with
         Address => Count'Address,
         Import;
      Temp_local_9 : aliased corecrt_h.size_t with
         Address => Offset'Address,
         Import;
      Temp_local_11 : aliased driver_types_h.cudaMemcpyKind with
         Address => Kind'Address,
         Import;

   begin
      declare

         Temp_res_1 : Integer :=
           Integer
             (cuda_runtime_api_h.cudaMemcpyFromSymbol
                (Temp_local_3, Temp_local_5, Temp_local_7, Temp_local_9,
                 Temp_local_11));

      begin
         null;

         declare
         begin
            if Temp_res_1 /= 0 then
               Ada.Exceptions.Raise_Exception
                 (CUDA.Exception_Registry.Element (Integer (Temp_res_1)));
            end if;
         end;
      end;
   end Memcpy_From_Symbol;

   procedure Memcpy_Async
     (Dst  : System.Address; Src : System.Address; Count : CUDA.Corecrt.Size_T;
      Kind : CUDA.Driver_Types.Memcpy_Kind; Stream : CUDA.Driver_Types.Stream_T)
   is

      Temp_local_3 : aliased System.Address with
         Address => Dst'Address,
         Import;
      Temp_local_5 : aliased System.Address with
         Address => Src'Address,
         Import;
      Temp_local_7 : aliased corecrt_h.size_t with
         Address => Count'Address,
         Import;
      Temp_local_9 : aliased driver_types_h.cudaMemcpyKind with
         Address => Kind'Address,
         Import;
      Temp_local_11 : aliased driver_types_h.cudaStream_t with
         Address => Stream'Address,
         Import;

   begin
      declare

         Temp_res_1 : Integer :=
           Integer
             (cuda_runtime_api_h.cudaMemcpyAsync
                (Temp_local_3, Temp_local_5, Temp_local_7, Temp_local_9,
                 Temp_local_11));

      begin
         null;

         declare
         begin
            if Temp_res_1 /= 0 then
               Ada.Exceptions.Raise_Exception
                 (CUDA.Exception_Registry.Element (Integer (Temp_res_1)));
            end if;
         end;
      end;
   end Memcpy_Async;

   procedure Memcpy_Peer_Async
     (Dst        : System.Address; Dst_Device : int; Src : System.Address;
      Src_Device : int; Count : CUDA.Corecrt.Size_T;
      Stream     : CUDA.Driver_Types.Stream_T)
   is

      Temp_local_3 : aliased System.Address with
         Address => Dst'Address,
         Import;
      Temp_local_5 : aliased int with
         Address => Dst_Device'Address,
         Import;
      Temp_local_7 : aliased System.Address with
         Address => Src'Address,
         Import;
      Temp_local_9 : aliased int with
         Address => Src_Device'Address,
         Import;
      Temp_local_11 : aliased corecrt_h.size_t with
         Address => Count'Address,
         Import;
      Temp_local_13 : aliased driver_types_h.cudaStream_t with
         Address => Stream'Address,
         Import;

   begin
      declare

         Temp_res_1 : Integer :=
           Integer
             (cuda_runtime_api_h.cudaMemcpyPeerAsync
                (Temp_local_3, Temp_local_5, Temp_local_7, Temp_local_9,
                 Temp_local_11, Temp_local_13));

      begin
         null;

         declare
         begin
            if Temp_res_1 /= 0 then
               Ada.Exceptions.Raise_Exception
                 (CUDA.Exception_Registry.Element (Integer (Temp_res_1)));
            end if;
         end;
      end;
   end Memcpy_Peer_Async;

   procedure Memcpy2_DAsync
     (Dst : System.Address; Dpitch : CUDA.Corecrt.Size_T; Src : System.Address;
      Spitch : CUDA.Corecrt.Size_T; Width : CUDA.Corecrt.Size_T;
      Height : CUDA.Corecrt.Size_T; Kind : CUDA.Driver_Types.Memcpy_Kind;
      Stream : CUDA.Driver_Types.Stream_T)
   is

      Temp_local_3 : aliased System.Address with
         Address => Dst'Address,
         Import;
      Temp_local_5 : aliased corecrt_h.size_t with
         Address => Dpitch'Address,
         Import;
      Temp_local_7 : aliased System.Address with
         Address => Src'Address,
         Import;
      Temp_local_9 : aliased corecrt_h.size_t with
         Address => Spitch'Address,
         Import;
      Temp_local_11 : aliased corecrt_h.size_t with
         Address => Width'Address,
         Import;
      Temp_local_13 : aliased corecrt_h.size_t with
         Address => Height'Address,
         Import;
      Temp_local_15 : aliased driver_types_h.cudaMemcpyKind with
         Address => Kind'Address,
         Import;
      Temp_local_17 : aliased driver_types_h.cudaStream_t with
         Address => Stream'Address,
         Import;

   begin
      declare

         Temp_res_1 : Integer :=
           Integer
             (cuda_runtime_api_h.cudaMemcpy2DAsync
                (Temp_local_3, Temp_local_5, Temp_local_7, Temp_local_9,
                 Temp_local_11, Temp_local_13, Temp_local_15, Temp_local_17));

      begin
         null;

         declare
         begin
            if Temp_res_1 /= 0 then
               Ada.Exceptions.Raise_Exception
                 (CUDA.Exception_Registry.Element (Integer (Temp_res_1)));
            end if;
         end;
      end;
   end Memcpy2_DAsync;

   procedure Memcpy2_DTo_Array_Async
     (Dst      : CUDA.Driver_Types.CUDA_Array_t; W_Offset : CUDA.Corecrt.Size_T;
      H_Offset : CUDA.Corecrt.Size_T; Src : System.Address;
      Spitch   : CUDA.Corecrt.Size_T; Width : CUDA.Corecrt.Size_T;
      Height   : CUDA.Corecrt.Size_T; Kind : CUDA.Driver_Types.Memcpy_Kind;
      Stream   : CUDA.Driver_Types.Stream_T)
   is

      Temp_local_3 : aliased driver_types_h.cudaArray_t with
         Address => Dst'Address,
         Import;
      Temp_local_5 : aliased corecrt_h.size_t with
         Address => W_Offset'Address,
         Import;
      Temp_local_7 : aliased corecrt_h.size_t with
         Address => H_Offset'Address,
         Import;
      Temp_local_9 : aliased System.Address with
         Address => Src'Address,
         Import;
      Temp_local_11 : aliased corecrt_h.size_t with
         Address => Spitch'Address,
         Import;
      Temp_local_13 : aliased corecrt_h.size_t with
         Address => Width'Address,
         Import;
      Temp_local_15 : aliased corecrt_h.size_t with
         Address => Height'Address,
         Import;
      Temp_local_17 : aliased driver_types_h.cudaMemcpyKind with
         Address => Kind'Address,
         Import;
      Temp_local_19 : aliased driver_types_h.cudaStream_t with
         Address => Stream'Address,
         Import;

   begin
      declare

         Temp_res_1 : Integer :=
           Integer
             (cuda_runtime_api_h.cudaMemcpy2DToArrayAsync
                (Temp_local_3, Temp_local_5, Temp_local_7, Temp_local_9,
                 Temp_local_11, Temp_local_13, Temp_local_15, Temp_local_17,
                 Temp_local_19));

      begin
         null;

         declare
         begin
            if Temp_res_1 /= 0 then
               Ada.Exceptions.Raise_Exception
                 (CUDA.Exception_Registry.Element (Integer (Temp_res_1)));
            end if;
         end;
      end;
   end Memcpy2_DTo_Array_Async;

   procedure Memcpy2_DFrom_Array_Async
     (Dst      : System.Address; Dpitch : CUDA.Corecrt.Size_T;
      Src      : CUDA.Driver_Types.CUDA_Array_const_t;
      W_Offset : CUDA.Corecrt.Size_T; H_Offset : CUDA.Corecrt.Size_T;
      Width    : CUDA.Corecrt.Size_T; Height : CUDA.Corecrt.Size_T;
      Kind : CUDA.Driver_Types.Memcpy_Kind; Stream : CUDA.Driver_Types.Stream_T)
   is

      Temp_local_3 : aliased System.Address with
         Address => Dst'Address,
         Import;
      Temp_local_5 : aliased corecrt_h.size_t with
         Address => Dpitch'Address,
         Import;
      Temp_local_7 : aliased driver_types_h.cudaArray_const_t with
         Address => Src'Address,
         Import;
      Temp_local_9 : aliased corecrt_h.size_t with
         Address => W_Offset'Address,
         Import;
      Temp_local_11 : aliased corecrt_h.size_t with
         Address => H_Offset'Address,
         Import;
      Temp_local_13 : aliased corecrt_h.size_t with
         Address => Width'Address,
         Import;
      Temp_local_15 : aliased corecrt_h.size_t with
         Address => Height'Address,
         Import;
      Temp_local_17 : aliased driver_types_h.cudaMemcpyKind with
         Address => Kind'Address,
         Import;
      Temp_local_19 : aliased driver_types_h.cudaStream_t with
         Address => Stream'Address,
         Import;

   begin
      declare

         Temp_res_1 : Integer :=
           Integer
             (cuda_runtime_api_h.cudaMemcpy2DFromArrayAsync
                (Temp_local_3, Temp_local_5, Temp_local_7, Temp_local_9,
                 Temp_local_11, Temp_local_13, Temp_local_15, Temp_local_17,
                 Temp_local_19));

      begin
         null;

         declare
         begin
            if Temp_res_1 /= 0 then
               Ada.Exceptions.Raise_Exception
                 (CUDA.Exception_Registry.Element (Integer (Temp_res_1)));
            end if;
         end;
      end;
   end Memcpy2_DFrom_Array_Async;

   procedure Memcpy_To_Symbol_Async
     (Symbol : System.Address; Src : System.Address;
      Count  : CUDA.Corecrt.Size_T; Offset : CUDA.Corecrt.Size_T;
      Kind : CUDA.Driver_Types.Memcpy_Kind; Stream : CUDA.Driver_Types.Stream_T)
   is

      Temp_local_3 : aliased System.Address with
         Address => Symbol'Address,
         Import;
      Temp_local_5 : aliased System.Address with
         Address => Src'Address,
         Import;
      Temp_local_7 : aliased corecrt_h.size_t with
         Address => Count'Address,
         Import;
      Temp_local_9 : aliased corecrt_h.size_t with
         Address => Offset'Address,
         Import;
      Temp_local_11 : aliased driver_types_h.cudaMemcpyKind with
         Address => Kind'Address,
         Import;
      Temp_local_13 : aliased driver_types_h.cudaStream_t with
         Address => Stream'Address,
         Import;

   begin
      declare

         Temp_res_1 : Integer :=
           Integer
             (cuda_runtime_api_h.cudaMemcpyToSymbolAsync
                (Temp_local_3, Temp_local_5, Temp_local_7, Temp_local_9,
                 Temp_local_11, Temp_local_13));

      begin
         null;

         declare
         begin
            if Temp_res_1 /= 0 then
               Ada.Exceptions.Raise_Exception
                 (CUDA.Exception_Registry.Element (Integer (Temp_res_1)));
            end if;
         end;
      end;
   end Memcpy_To_Symbol_Async;

   procedure Memcpy_From_Symbol_Async
     (Dst   : System.Address; Symbol : System.Address;
      Count : CUDA.Corecrt.Size_T; Offset : CUDA.Corecrt.Size_T;
      Kind : CUDA.Driver_Types.Memcpy_Kind; Stream : CUDA.Driver_Types.Stream_T)
   is

      Temp_local_3 : aliased System.Address with
         Address => Dst'Address,
         Import;
      Temp_local_5 : aliased System.Address with
         Address => Symbol'Address,
         Import;
      Temp_local_7 : aliased corecrt_h.size_t with
         Address => Count'Address,
         Import;
      Temp_local_9 : aliased corecrt_h.size_t with
         Address => Offset'Address,
         Import;
      Temp_local_11 : aliased driver_types_h.cudaMemcpyKind with
         Address => Kind'Address,
         Import;
      Temp_local_13 : aliased driver_types_h.cudaStream_t with
         Address => Stream'Address,
         Import;

   begin
      declare

         Temp_res_1 : Integer :=
           Integer
             (cuda_runtime_api_h.cudaMemcpyFromSymbolAsync
                (Temp_local_3, Temp_local_5, Temp_local_7, Temp_local_9,
                 Temp_local_11, Temp_local_13));

      begin
         null;

         declare
         begin
            if Temp_res_1 /= 0 then
               Ada.Exceptions.Raise_Exception
                 (CUDA.Exception_Registry.Element (Integer (Temp_res_1)));
            end if;
         end;
      end;
   end Memcpy_From_Symbol_Async;

   procedure Memset
     (Dev_Ptr : System.Address; Value : int; Count : CUDA.Corecrt.Size_T)
   is

      Temp_local_3 : aliased System.Address with
         Address => Dev_Ptr'Address,
         Import;
      Temp_local_5 : aliased int with
         Address => Value'Address,
         Import;
      Temp_local_7 : aliased corecrt_h.size_t with
         Address => Count'Address,
         Import;

   begin
      declare

         Temp_res_1 : Integer :=
           Integer
             (cuda_runtime_api_h.cudaMemset
                (Temp_local_3, Temp_local_5, Temp_local_7));

      begin
         null;

         declare
         begin
            if Temp_res_1 /= 0 then
               Ada.Exceptions.Raise_Exception
                 (CUDA.Exception_Registry.Element (Integer (Temp_res_1)));
            end if;
         end;
      end;
   end Memset;

   procedure Memset2_D
     (Dev_Ptr : System.Address; Pitch : CUDA.Corecrt.Size_T; Value : int;
      Width   : CUDA.Corecrt.Size_T; Height : CUDA.Corecrt.Size_T)
   is

      Temp_local_3 : aliased System.Address with
         Address => Dev_Ptr'Address,
         Import;
      Temp_local_5 : aliased corecrt_h.size_t with
         Address => Pitch'Address,
         Import;
      Temp_local_7 : aliased int with
         Address => Value'Address,
         Import;
      Temp_local_9 : aliased corecrt_h.size_t with
         Address => Width'Address,
         Import;
      Temp_local_11 : aliased corecrt_h.size_t with
         Address => Height'Address,
         Import;

   begin
      declare

         Temp_res_1 : Integer :=
           Integer
             (cuda_runtime_api_h.cudaMemset2D
                (Temp_local_3, Temp_local_5, Temp_local_7, Temp_local_9,
                 Temp_local_11));

      begin
         null;

         declare
         begin
            if Temp_res_1 /= 0 then
               Ada.Exceptions.Raise_Exception
                 (CUDA.Exception_Registry.Element (Integer (Temp_res_1)));
            end if;
         end;
      end;
   end Memset2_D;

   procedure Memset3_D
     (Pitched_Dev_Ptr : CUDA.Driver_Types.Pitched_Ptr; Value : int;
      Extent          : CUDA.Driver_Types.Extent_T)
   is

      Temp_local_3 : aliased driver_types_h.cudaPitchedPtr with
         Address => Pitched_Dev_Ptr'Address,
         Import;
      Temp_local_5 : aliased int with
         Address => Value'Address,
         Import;
      Temp_local_7 : aliased driver_types_h.cudaExtent with
         Address => Extent'Address,
         Import;

   begin
      declare

         Temp_res_1 : Integer :=
           Integer
             (cuda_runtime_api_h.cudaMemset3D
                (Temp_local_3, Temp_local_5, Temp_local_7));

      begin
         null;

         declare
         begin
            if Temp_res_1 /= 0 then
               Ada.Exceptions.Raise_Exception
                 (CUDA.Exception_Registry.Element (Integer (Temp_res_1)));
            end if;
         end;
      end;
   end Memset3_D;

   procedure Memset_Async
     (Dev_Ptr : System.Address; Value : int; Count : CUDA.Corecrt.Size_T;
      Stream  : CUDA.Driver_Types.Stream_T)
   is

      Temp_local_3 : aliased System.Address with
         Address => Dev_Ptr'Address,
         Import;
      Temp_local_5 : aliased int with
         Address => Value'Address,
         Import;
      Temp_local_7 : aliased corecrt_h.size_t with
         Address => Count'Address,
         Import;
      Temp_local_9 : aliased driver_types_h.cudaStream_t with
         Address => Stream'Address,
         Import;

   begin
      declare

         Temp_res_1 : Integer :=
           Integer
             (cuda_runtime_api_h.cudaMemsetAsync
                (Temp_local_3, Temp_local_5, Temp_local_7, Temp_local_9));

      begin
         null;

         declare
         begin
            if Temp_res_1 /= 0 then
               Ada.Exceptions.Raise_Exception
                 (CUDA.Exception_Registry.Element (Integer (Temp_res_1)));
            end if;
         end;
      end;
   end Memset_Async;

   procedure Memset2_DAsync
     (Dev_Ptr : System.Address; Pitch : CUDA.Corecrt.Size_T; Value : int;
      Width   : CUDA.Corecrt.Size_T; Height : CUDA.Corecrt.Size_T;
      Stream  : CUDA.Driver_Types.Stream_T)
   is

      Temp_local_3 : aliased System.Address with
         Address => Dev_Ptr'Address,
         Import;
      Temp_local_5 : aliased corecrt_h.size_t with
         Address => Pitch'Address,
         Import;
      Temp_local_7 : aliased int with
         Address => Value'Address,
         Import;
      Temp_local_9 : aliased corecrt_h.size_t with
         Address => Width'Address,
         Import;
      Temp_local_11 : aliased corecrt_h.size_t with
         Address => Height'Address,
         Import;
      Temp_local_13 : aliased driver_types_h.cudaStream_t with
         Address => Stream'Address,
         Import;

   begin
      declare

         Temp_res_1 : Integer :=
           Integer
             (cuda_runtime_api_h.cudaMemset2DAsync
                (Temp_local_3, Temp_local_5, Temp_local_7, Temp_local_9,
                 Temp_local_11, Temp_local_13));

      begin
         null;

         declare
         begin
            if Temp_res_1 /= 0 then
               Ada.Exceptions.Raise_Exception
                 (CUDA.Exception_Registry.Element (Integer (Temp_res_1)));
            end if;
         end;
      end;
   end Memset2_DAsync;

   procedure Memset3_DAsync
     (Pitched_Dev_Ptr : CUDA.Driver_Types.Pitched_Ptr; Value : int;
      Extent : CUDA.Driver_Types.Extent_T; Stream : CUDA.Driver_Types.Stream_T)
   is

      Temp_local_3 : aliased driver_types_h.cudaPitchedPtr with
         Address => Pitched_Dev_Ptr'Address,
         Import;
      Temp_local_5 : aliased int with
         Address => Value'Address,
         Import;
      Temp_local_7 : aliased driver_types_h.cudaExtent with
         Address => Extent'Address,
         Import;
      Temp_local_9 : aliased driver_types_h.cudaStream_t with
         Address => Stream'Address,
         Import;

   begin
      declare

         Temp_res_1 : Integer :=
           Integer
             (cuda_runtime_api_h.cudaMemset3DAsync
                (Temp_local_3, Temp_local_5, Temp_local_7, Temp_local_9));

      begin
         null;

         declare
         begin
            if Temp_res_1 /= 0 then
               Ada.Exceptions.Raise_Exception
                 (CUDA.Exception_Registry.Element (Integer (Temp_res_1)));
            end if;
         end;
      end;
   end Memset3_DAsync;

   procedure Get_Symbol_Address
     (Dev_Ptr : System.Address; Symbol : System.Address)
   is

      Temp_local_3 : aliased System.Address with
         Address => Dev_Ptr'Address,
         Import;
      Temp_local_5 : aliased System.Address with
         Address => Symbol'Address,
         Import;

   begin
      declare

         Temp_res_1 : Integer :=
           Integer
             (cuda_runtime_api_h.cudaGetSymbolAddress
                (Temp_local_3, Temp_local_5));

      begin
         null;

         declare
         begin
            if Temp_res_1 /= 0 then
               Ada.Exceptions.Raise_Exception
                 (CUDA.Exception_Registry.Element (Integer (Temp_res_1)));
            end if;
         end;
      end;
   end Get_Symbol_Address;

   function Get_Symbol_Size (Symbol : System.Address) return CUDA.Corecrt.Size_T
   is

      Temp_call_1 : aliased corecrt_h.size_t;
      Temp_ret_2  : aliased CUDA.Corecrt.Size_T with
         Address => Temp_call_1'Address,
         Import;
      Temp_local_4 : aliased System.Address with
         Address => Symbol'Address,
         Import;

   begin
      declare

         Temp_res_1 : Integer :=
           Integer
             (cuda_runtime_api_h.cudaGetSymbolSize
                (Temp_call_1'Unchecked_Access, Temp_local_4));

      begin
         null;

         declare
         begin
            if Temp_res_1 /= 0 then
               Ada.Exceptions.Raise_Exception
                 (CUDA.Exception_Registry.Element (Integer (Temp_res_1)));
            end if;
            return Temp_ret_2;
         end;
      end;
   end Get_Symbol_Size;

   procedure Mem_Prefetch_Async
     (Dev_Ptr : System.Address; Count : CUDA.Corecrt.Size_T; Dst_Device : int;
      Stream  : CUDA.Driver_Types.Stream_T)
   is

      Temp_local_3 : aliased System.Address with
         Address => Dev_Ptr'Address,
         Import;
      Temp_local_5 : aliased corecrt_h.size_t with
         Address => Count'Address,
         Import;
      Temp_local_7 : aliased int with
         Address => Dst_Device'Address,
         Import;
      Temp_local_9 : aliased driver_types_h.cudaStream_t with
         Address => Stream'Address,
         Import;

   begin
      declare

         Temp_res_1 : Integer :=
           Integer
             (cuda_runtime_api_h.cudaMemPrefetchAsync
                (Temp_local_3, Temp_local_5, Temp_local_7, Temp_local_9));

      begin
         null;

         declare
         begin
            if Temp_res_1 /= 0 then
               Ada.Exceptions.Raise_Exception
                 (CUDA.Exception_Registry.Element (Integer (Temp_res_1)));
            end if;
         end;
      end;
   end Mem_Prefetch_Async;

   procedure Mem_Advise
     (Dev_Ptr : System.Address; Count : CUDA.Corecrt.Size_T;
      Advice  : CUDA.Driver_Types.Memory_Advise; Device : int)
   is

      Temp_local_3 : aliased System.Address with
         Address => Dev_Ptr'Address,
         Import;
      Temp_local_5 : aliased corecrt_h.size_t with
         Address => Count'Address,
         Import;
      Temp_local_7 : aliased driver_types_h.cudaMemoryAdvise with
         Address => Advice'Address,
         Import;
      Temp_local_9 : aliased int with
         Address => Device'Address,
         Import;

   begin
      declare

         Temp_res_1 : Integer :=
           Integer
             (cuda_runtime_api_h.cudaMemAdvise
                (Temp_local_3, Temp_local_5, Temp_local_7, Temp_local_9));

      begin
         null;

         declare
         begin
            if Temp_res_1 /= 0 then
               Ada.Exceptions.Raise_Exception
                 (CUDA.Exception_Registry.Element (Integer (Temp_res_1)));
            end if;
         end;
      end;
   end Mem_Advise;

   procedure Mem_Range_Get_Attribute
     (Data      : System.Address; Data_Size : CUDA.Corecrt.Size_T;
      Attribute : CUDA.Driver_Types.Mem_Range_Attribute;
      Dev_Ptr   : System.Address; Count : CUDA.Corecrt.Size_T)
   is

      Temp_local_3 : aliased System.Address with
         Address => Data'Address,
         Import;
      Temp_local_5 : aliased corecrt_h.size_t with
         Address => Data_Size'Address,
         Import;
      Temp_local_7 : aliased driver_types_h.cudaMemRangeAttribute with
         Address => Attribute'Address,
         Import;
      Temp_local_9 : aliased System.Address with
         Address => Dev_Ptr'Address,
         Import;
      Temp_local_11 : aliased corecrt_h.size_t with
         Address => Count'Address,
         Import;

   begin
      declare

         Temp_res_1 : Integer :=
           Integer
             (cuda_runtime_api_h.cudaMemRangeGetAttribute
                (Temp_local_3, Temp_local_5, Temp_local_7, Temp_local_9,
                 Temp_local_11));

      begin
         null;

         declare
         begin
            if Temp_res_1 /= 0 then
               Ada.Exceptions.Raise_Exception
                 (CUDA.Exception_Registry.Element (Integer (Temp_res_1)));
            end if;
         end;
      end;
   end Mem_Range_Get_Attribute;

   procedure Mem_Range_Get_Attributes
     (Data           :     System.Address; Data_Sizes : out CUDA.Corecrt.Size_T;
      Attributes     : out CUDA.Driver_Types.Mem_Range_Attribute;
      Num_Attributes :     CUDA.Corecrt.Size_T; Dev_Ptr : System.Address;
      Count          :     CUDA.Corecrt.Size_T)
   is

      Temp_local_3 : aliased System.Address with
         Address => Data'Address,
         Import;
      Temp_call_4 : aliased corecrt_h.size_t with
         Address => Data_Sizes'Address,
         Import;
      Temp_call_6 : aliased driver_types_h.cudaMemRangeAttribute with
         Address => Attributes'Address,
         Import;
      Temp_local_9 : aliased corecrt_h.size_t with
         Address => Num_Attributes'Address,
         Import;
      Temp_local_11 : aliased System.Address with
         Address => Dev_Ptr'Address,
         Import;
      Temp_local_13 : aliased corecrt_h.size_t with
         Address => Count'Address,
         Import;

   begin
      declare

         Temp_res_1 : Integer :=
           Integer
             (cuda_runtime_api_h.cudaMemRangeGetAttributes
                (Temp_local_3, Temp_call_4'Unchecked_Access,
                 Temp_call_6'Unchecked_Access, Temp_local_9, Temp_local_11,
                 Temp_local_13));

      begin
         null;

         declare
         begin
            if Temp_res_1 /= 0 then
               Ada.Exceptions.Raise_Exception
                 (CUDA.Exception_Registry.Element (Integer (Temp_res_1)));
            end if;
         end;
      end;
   end Mem_Range_Get_Attributes;

   procedure Memcpy_To_Array
     (Dst      : CUDA.Driver_Types.CUDA_Array_t; W_Offset : CUDA.Corecrt.Size_T;
      H_Offset : CUDA.Corecrt.Size_T; Src : System.Address;
      Count    : CUDA.Corecrt.Size_T; Kind : CUDA.Driver_Types.Memcpy_Kind)
   is

      Temp_local_3 : aliased driver_types_h.cudaArray_t with
         Address => Dst'Address,
         Import;
      Temp_local_5 : aliased corecrt_h.size_t with
         Address => W_Offset'Address,
         Import;
      Temp_local_7 : aliased corecrt_h.size_t with
         Address => H_Offset'Address,
         Import;
      Temp_local_9 : aliased System.Address with
         Address => Src'Address,
         Import;
      Temp_local_11 : aliased corecrt_h.size_t with
         Address => Count'Address,
         Import;
      Temp_local_13 : aliased driver_types_h.cudaMemcpyKind with
         Address => Kind'Address,
         Import;

   begin
      declare

         Temp_res_1 : Integer :=
           Integer
             (cuda_runtime_api_h.cudaMemcpyToArray
                (Temp_local_3, Temp_local_5, Temp_local_7, Temp_local_9,
                 Temp_local_11, Temp_local_13));

      begin
         null;

         declare
         begin
            if Temp_res_1 /= 0 then
               Ada.Exceptions.Raise_Exception
                 (CUDA.Exception_Registry.Element (Integer (Temp_res_1)));
            end if;
         end;
      end;
   end Memcpy_To_Array;

   procedure Memcpy_From_Array
     (Dst      : System.Address; Src : CUDA.Driver_Types.CUDA_Array_const_t;
      W_Offset : CUDA.Corecrt.Size_T; H_Offset : CUDA.Corecrt.Size_T;
      Count    : CUDA.Corecrt.Size_T; Kind : CUDA.Driver_Types.Memcpy_Kind)
   is

      Temp_local_3 : aliased System.Address with
         Address => Dst'Address,
         Import;
      Temp_local_5 : aliased driver_types_h.cudaArray_const_t with
         Address => Src'Address,
         Import;
      Temp_local_7 : aliased corecrt_h.size_t with
         Address => W_Offset'Address,
         Import;
      Temp_local_9 : aliased corecrt_h.size_t with
         Address => H_Offset'Address,
         Import;
      Temp_local_11 : aliased corecrt_h.size_t with
         Address => Count'Address,
         Import;
      Temp_local_13 : aliased driver_types_h.cudaMemcpyKind with
         Address => Kind'Address,
         Import;

   begin
      declare

         Temp_res_1 : Integer :=
           Integer
             (cuda_runtime_api_h.cudaMemcpyFromArray
                (Temp_local_3, Temp_local_5, Temp_local_7, Temp_local_9,
                 Temp_local_11, Temp_local_13));

      begin
         null;

         declare
         begin
            if Temp_res_1 /= 0 then
               Ada.Exceptions.Raise_Exception
                 (CUDA.Exception_Registry.Element (Integer (Temp_res_1)));
            end if;
         end;
      end;
   end Memcpy_From_Array;

   procedure Memcpy_Array_To_Array
     (Dst : CUDA.Driver_Types.CUDA_Array_t; W_Offset_Dst : CUDA.Corecrt.Size_T;
      H_Offset_Dst : CUDA.Corecrt.Size_T;
      Src          : CUDA.Driver_Types.CUDA_Array_const_t;
      W_Offset_Src : CUDA.Corecrt.Size_T; H_Offset_Src : CUDA.Corecrt.Size_T;
      Count        : CUDA.Corecrt.Size_T; Kind : CUDA.Driver_Types.Memcpy_Kind)
   is

      Temp_local_3 : aliased driver_types_h.cudaArray_t with
         Address => Dst'Address,
         Import;
      Temp_local_5 : aliased corecrt_h.size_t with
         Address => W_Offset_Dst'Address,
         Import;
      Temp_local_7 : aliased corecrt_h.size_t with
         Address => H_Offset_Dst'Address,
         Import;
      Temp_local_9 : aliased driver_types_h.cudaArray_const_t with
         Address => Src'Address,
         Import;
      Temp_local_11 : aliased corecrt_h.size_t with
         Address => W_Offset_Src'Address,
         Import;
      Temp_local_13 : aliased corecrt_h.size_t with
         Address => H_Offset_Src'Address,
         Import;
      Temp_local_15 : aliased corecrt_h.size_t with
         Address => Count'Address,
         Import;
      Temp_local_17 : aliased driver_types_h.cudaMemcpyKind with
         Address => Kind'Address,
         Import;

   begin
      declare

         Temp_res_1 : Integer :=
           Integer
             (cuda_runtime_api_h.cudaMemcpyArrayToArray
                (Temp_local_3, Temp_local_5, Temp_local_7, Temp_local_9,
                 Temp_local_11, Temp_local_13, Temp_local_15, Temp_local_17));

      begin
         null;

         declare
         begin
            if Temp_res_1 /= 0 then
               Ada.Exceptions.Raise_Exception
                 (CUDA.Exception_Registry.Element (Integer (Temp_res_1)));
            end if;
         end;
      end;
   end Memcpy_Array_To_Array;

   procedure Memcpy_To_Array_Async
     (Dst      : CUDA.Driver_Types.CUDA_Array_t; W_Offset : CUDA.Corecrt.Size_T;
      H_Offset : CUDA.Corecrt.Size_T; Src : System.Address;
      Count    : CUDA.Corecrt.Size_T; Kind : CUDA.Driver_Types.Memcpy_Kind;
      Stream   : CUDA.Driver_Types.Stream_T)
   is

      Temp_local_3 : aliased driver_types_h.cudaArray_t with
         Address => Dst'Address,
         Import;
      Temp_local_5 : aliased corecrt_h.size_t with
         Address => W_Offset'Address,
         Import;
      Temp_local_7 : aliased corecrt_h.size_t with
         Address => H_Offset'Address,
         Import;
      Temp_local_9 : aliased System.Address with
         Address => Src'Address,
         Import;
      Temp_local_11 : aliased corecrt_h.size_t with
         Address => Count'Address,
         Import;
      Temp_local_13 : aliased driver_types_h.cudaMemcpyKind with
         Address => Kind'Address,
         Import;
      Temp_local_15 : aliased driver_types_h.cudaStream_t with
         Address => Stream'Address,
         Import;

   begin
      declare

         Temp_res_1 : Integer :=
           Integer
             (cuda_runtime_api_h.cudaMemcpyToArrayAsync
                (Temp_local_3, Temp_local_5, Temp_local_7, Temp_local_9,
                 Temp_local_11, Temp_local_13, Temp_local_15));

      begin
         null;

         declare
         begin
            if Temp_res_1 /= 0 then
               Ada.Exceptions.Raise_Exception
                 (CUDA.Exception_Registry.Element (Integer (Temp_res_1)));
            end if;
         end;
      end;
   end Memcpy_To_Array_Async;

   procedure Memcpy_From_Array_Async
     (Dst      : System.Address; Src : CUDA.Driver_Types.CUDA_Array_const_t;
      W_Offset : CUDA.Corecrt.Size_T; H_Offset : CUDA.Corecrt.Size_T;
      Count    : CUDA.Corecrt.Size_T; Kind : CUDA.Driver_Types.Memcpy_Kind;
      Stream   : CUDA.Driver_Types.Stream_T)
   is

      Temp_local_3 : aliased System.Address with
         Address => Dst'Address,
         Import;
      Temp_local_5 : aliased driver_types_h.cudaArray_const_t with
         Address => Src'Address,
         Import;
      Temp_local_7 : aliased corecrt_h.size_t with
         Address => W_Offset'Address,
         Import;
      Temp_local_9 : aliased corecrt_h.size_t with
         Address => H_Offset'Address,
         Import;
      Temp_local_11 : aliased corecrt_h.size_t with
         Address => Count'Address,
         Import;
      Temp_local_13 : aliased driver_types_h.cudaMemcpyKind with
         Address => Kind'Address,
         Import;
      Temp_local_15 : aliased driver_types_h.cudaStream_t with
         Address => Stream'Address,
         Import;

   begin
      declare

         Temp_res_1 : Integer :=
           Integer
             (cuda_runtime_api_h.cudaMemcpyFromArrayAsync
                (Temp_local_3, Temp_local_5, Temp_local_7, Temp_local_9,
                 Temp_local_11, Temp_local_13, Temp_local_15));

      begin
         null;

         declare
         begin
            if Temp_res_1 /= 0 then
               Ada.Exceptions.Raise_Exception
                 (CUDA.Exception_Registry.Element (Integer (Temp_res_1)));
            end if;
         end;
      end;
   end Memcpy_From_Array_Async;

   function Pointer_Get_Attributes
     (Ptr : System.Address) return CUDA.Driver_Types.Pointer_Attributes
   is

      Temp_call_1 : aliased driver_types_h.cudaPointerAttributes;
      Temp_ret_2  : aliased CUDA.Driver_Types.Pointer_Attributes with
         Address => Temp_call_1'Address,
         Import;
      Temp_local_4 : aliased System.Address with
         Address => Ptr'Address,
         Import;

   begin
      declare

         Temp_res_1 : Integer :=
           Integer
             (cuda_runtime_api_h.cudaPointerGetAttributes
                (Temp_call_1'Unchecked_Access, Temp_local_4));

      begin
         null;

         declare
         begin
            if Temp_res_1 /= 0 then
               Ada.Exceptions.Raise_Exception
                 (CUDA.Exception_Registry.Element (Integer (Temp_res_1)));
            end if;
            return Temp_ret_2;
         end;
      end;
   end Pointer_Get_Attributes;

   procedure Device_Can_Access_Peer
     (Can_Access_Peer : out int; Device : int; Peer_Device : int)
   is

      Temp_call_2 : aliased int with
         Address => Can_Access_Peer'Address,
         Import;
      Temp_local_5 : aliased int with
         Address => Device'Address,
         Import;
      Temp_local_7 : aliased int with
         Address => Peer_Device'Address,
         Import;

   begin
      declare

         Temp_res_1 : Integer :=
           Integer
             (cuda_runtime_api_h.cudaDeviceCanAccessPeer
                (Temp_call_2'Unchecked_Access, Temp_local_5, Temp_local_7));

      begin
         null;

         declare
         begin
            if Temp_res_1 /= 0 then
               Ada.Exceptions.Raise_Exception
                 (CUDA.Exception_Registry.Element (Integer (Temp_res_1)));
            end if;
         end;
      end;
   end Device_Can_Access_Peer;

   procedure Device_Enable_Peer_Access (Peer_Device : int; Flags : unsigned) is

      Temp_local_3 : aliased int with
         Address => Peer_Device'Address,
         Import;
      Temp_local_5 : aliased unsigned with
         Address => Flags'Address,
         Import;

   begin
      declare

         Temp_res_1 : Integer :=
           Integer
             (cuda_runtime_api_h.cudaDeviceEnablePeerAccess
                (Temp_local_3, Temp_local_5));

      begin
         null;

         declare
         begin
            if Temp_res_1 /= 0 then
               Ada.Exceptions.Raise_Exception
                 (CUDA.Exception_Registry.Element (Integer (Temp_res_1)));
            end if;
         end;
      end;
   end Device_Enable_Peer_Access;

   procedure Device_Disable_Peer_Access (Peer_Device : int) is

      Temp_local_3 : aliased int with
         Address => Peer_Device'Address,
         Import;

   begin
      declare

         Temp_res_1 : Integer :=
           Integer
             (cuda_runtime_api_h.cudaDeviceDisablePeerAccess (Temp_local_3));

      begin
         null;

         declare
         begin
            if Temp_res_1 /= 0 then
               Ada.Exceptions.Raise_Exception
                 (CUDA.Exception_Registry.Element (Integer (Temp_res_1)));
            end if;
         end;
      end;
   end Device_Disable_Peer_Access;

   procedure Graphics_Unregister_Resource
     (Resource : CUDA.Driver_Types.Graphics_Resource_T)
   is

      Temp_local_3 : aliased driver_types_h.cudaGraphicsResource_t with
         Address => Resource'Address,
         Import;

   begin
      declare

         Temp_res_1 : Integer :=
           Integer
             (cuda_runtime_api_h.cudaGraphicsUnregisterResource (Temp_local_3));

      begin
         null;

         declare
         begin
            if Temp_res_1 /= 0 then
               Ada.Exceptions.Raise_Exception
                 (CUDA.Exception_Registry.Element (Integer (Temp_res_1)));
            end if;
         end;
      end;
   end Graphics_Unregister_Resource;

   procedure Graphics_Resource_Set_Map_Flags
     (Resource : CUDA.Driver_Types.Graphics_Resource_T; Flags : unsigned)
   is

      Temp_local_3 : aliased driver_types_h.cudaGraphicsResource_t with
         Address => Resource'Address,
         Import;
      Temp_local_5 : aliased unsigned with
         Address => Flags'Address,
         Import;

   begin
      declare

         Temp_res_1 : Integer :=
           Integer
             (cuda_runtime_api_h.cudaGraphicsResourceSetMapFlags
                (Temp_local_3, Temp_local_5));

      begin
         null;

         declare
         begin
            if Temp_res_1 /= 0 then
               Ada.Exceptions.Raise_Exception
                 (CUDA.Exception_Registry.Element (Integer (Temp_res_1)));
            end if;
         end;
      end;
   end Graphics_Resource_Set_Map_Flags;

   procedure Graphics_Map_Resources
     (Count  : int; Resources : System.Address;
      Stream : CUDA.Driver_Types.Stream_T)
   is

      Temp_local_3 : aliased int with
         Address => Count'Address,
         Import;
      Temp_local_5 : aliased System.Address with
         Address => Resources'Address,
         Import;
      Temp_local_7 : aliased driver_types_h.cudaStream_t with
         Address => Stream'Address,
         Import;

   begin
      declare

         Temp_res_1 : Integer :=
           Integer
             (cuda_runtime_api_h.cudaGraphicsMapResources
                (Temp_local_3, Temp_local_5, Temp_local_7));

      begin
         null;

         declare
         begin
            if Temp_res_1 /= 0 then
               Ada.Exceptions.Raise_Exception
                 (CUDA.Exception_Registry.Element (Integer (Temp_res_1)));
            end if;
         end;
      end;
   end Graphics_Map_Resources;

   procedure Graphics_Unmap_Resources
     (Count  : int; Resources : System.Address;
      Stream : CUDA.Driver_Types.Stream_T)
   is

      Temp_local_3 : aliased int with
         Address => Count'Address,
         Import;
      Temp_local_5 : aliased System.Address with
         Address => Resources'Address,
         Import;
      Temp_local_7 : aliased driver_types_h.cudaStream_t with
         Address => Stream'Address,
         Import;

   begin
      declare

         Temp_res_1 : Integer :=
           Integer
             (cuda_runtime_api_h.cudaGraphicsUnmapResources
                (Temp_local_3, Temp_local_5, Temp_local_7));

      begin
         null;

         declare
         begin
            if Temp_res_1 /= 0 then
               Ada.Exceptions.Raise_Exception
                 (CUDA.Exception_Registry.Element (Integer (Temp_res_1)));
            end if;
         end;
      end;
   end Graphics_Unmap_Resources;

   procedure Graphics_Resource_Get_Mapped_Pointer
     (Dev_Ptr  : System.Address; Size : out CUDA.Corecrt.Size_T;
      Resource : CUDA.Driver_Types.Graphics_Resource_T)
   is

      Temp_local_3 : aliased System.Address with
         Address => Dev_Ptr'Address,
         Import;
      Temp_call_4 : aliased corecrt_h.size_t with
         Address => Size'Address,
         Import;
      Temp_local_7 : aliased driver_types_h.cudaGraphicsResource_t with
         Address => Resource'Address,
         Import;

   begin
      declare

         Temp_res_1 : Integer :=
           Integer
             (cuda_runtime_api_h.cudaGraphicsResourceGetMappedPointer
                (Temp_local_3, Temp_call_4'Unchecked_Access, Temp_local_7));

      begin
         null;

         declare
         begin
            if Temp_res_1 /= 0 then
               Ada.Exceptions.Raise_Exception
                 (CUDA.Exception_Registry.Element (Integer (Temp_res_1)));
            end if;
         end;
      end;
   end Graphics_Resource_Get_Mapped_Pointer;

   procedure Graphics_Sub_Resource_Get_Mapped_Array
     (C_Array   : System.Address;
      Resource  : CUDA.Driver_Types.Graphics_Resource_T; Array_Index : unsigned;
      Mip_Level : unsigned)
   is

      Temp_local_3 : aliased System.Address with
         Address => C_Array'Address,
         Import;
      Temp_local_5 : aliased driver_types_h.cudaGraphicsResource_t with
         Address => Resource'Address,
         Import;
      Temp_local_7 : aliased unsigned with
         Address => Array_Index'Address,
         Import;
      Temp_local_9 : aliased unsigned with
         Address => Mip_Level'Address,
         Import;

   begin
      declare

         Temp_res_1 : Integer :=
           Integer
             (cuda_runtime_api_h.cudaGraphicsSubResourceGetMappedArray
                (Temp_local_3, Temp_local_5, Temp_local_7, Temp_local_9));

      begin
         null;

         declare
         begin
            if Temp_res_1 /= 0 then
               Ada.Exceptions.Raise_Exception
                 (CUDA.Exception_Registry.Element (Integer (Temp_res_1)));
            end if;
         end;
      end;
   end Graphics_Sub_Resource_Get_Mapped_Array;

   procedure Graphics_Resource_Get_Mapped_Mipmapped_Array
     (Mipmapped_Array : System.Address;
      Resource        : CUDA.Driver_Types.Graphics_Resource_T)
   is

      Temp_local_3 : aliased System.Address with
         Address => Mipmapped_Array'Address,
         Import;
      Temp_local_5 : aliased driver_types_h.cudaGraphicsResource_t with
         Address => Resource'Address,
         Import;

   begin
      declare

         Temp_res_1 : Integer :=
           Integer
             (cuda_runtime_api_h.cudaGraphicsResourceGetMappedMipmappedArray
                (Temp_local_3, Temp_local_5));

      begin
         null;

         declare
         begin
            if Temp_res_1 /= 0 then
               Ada.Exceptions.Raise_Exception
                 (CUDA.Exception_Registry.Element (Integer (Temp_res_1)));
            end if;
         end;
      end;
   end Graphics_Resource_Get_Mapped_Mipmapped_Array;

   procedure Bind_Texture
     (Offset  : out CUDA.Corecrt.Size_T;
      Texref  : out CUDA.Texture_Types.Texture_Reference;
      Dev_Ptr :     System.Address;
      Desc    : out CUDA.Driver_Types.Channel_Format_Desc;
      Size    :     CUDA.Corecrt.Size_T)
   is

      Temp_call_2 : aliased corecrt_h.size_t with
         Address => Offset'Address,
         Import;
      Temp_call_4 : aliased texture_types_h.textureReference with
         Address => Texref'Address,
         Import;
      Temp_local_7 : aliased System.Address with
         Address => Dev_Ptr'Address,
         Import;
      Temp_call_8 : aliased driver_types_h.cudaChannelFormatDesc with
         Address => Desc'Address,
         Import;
      Temp_local_11 : aliased corecrt_h.size_t with
         Address => Size'Address,
         Import;

   begin
      declare

         Temp_res_1 : Integer :=
           Integer
             (cuda_runtime_api_h.cudaBindTexture
                (Temp_call_2'Unchecked_Access, Temp_call_4'Unchecked_Access,
                 Temp_local_7, Temp_call_8'Unchecked_Access, Temp_local_11));

      begin
         null;

         declare
         begin
            if Temp_res_1 /= 0 then
               Ada.Exceptions.Raise_Exception
                 (CUDA.Exception_Registry.Element (Integer (Temp_res_1)));
            end if;
         end;
      end;
   end Bind_Texture;

   procedure Bind_Texture2_D
     (Offset  : out CUDA.Corecrt.Size_T;
      Texref  : out CUDA.Texture_Types.Texture_Reference;
      Dev_Ptr :     System.Address;
      Desc    : out CUDA.Driver_Types.Channel_Format_Desc;
      Width   :     CUDA.Corecrt.Size_T; Height : CUDA.Corecrt.Size_T;
      Pitch   :     CUDA.Corecrt.Size_T)
   is

      Temp_call_2 : aliased corecrt_h.size_t with
         Address => Offset'Address,
         Import;
      Temp_call_4 : aliased texture_types_h.textureReference with
         Address => Texref'Address,
         Import;
      Temp_local_7 : aliased System.Address with
         Address => Dev_Ptr'Address,
         Import;
      Temp_call_8 : aliased driver_types_h.cudaChannelFormatDesc with
         Address => Desc'Address,
         Import;
      Temp_local_11 : aliased corecrt_h.size_t with
         Address => Width'Address,
         Import;
      Temp_local_13 : aliased corecrt_h.size_t with
         Address => Height'Address,
         Import;
      Temp_local_15 : aliased corecrt_h.size_t with
         Address => Pitch'Address,
         Import;

   begin
      declare

         Temp_res_1 : Integer :=
           Integer
             (cuda_runtime_api_h.cudaBindTexture2D
                (Temp_call_2'Unchecked_Access, Temp_call_4'Unchecked_Access,
                 Temp_local_7, Temp_call_8'Unchecked_Access, Temp_local_11,
                 Temp_local_13, Temp_local_15));

      begin
         null;

         declare
         begin
            if Temp_res_1 /= 0 then
               Ada.Exceptions.Raise_Exception
                 (CUDA.Exception_Registry.Element (Integer (Temp_res_1)));
            end if;
         end;
      end;
   end Bind_Texture2_D;

   procedure Bind_Texture_To_Array
     (Texref  : out CUDA.Texture_Types.Texture_Reference;
      C_Array :     CUDA.Driver_Types.CUDA_Array_const_t;
      Desc    : out CUDA.Driver_Types.Channel_Format_Desc)
   is

      Temp_call_2 : aliased texture_types_h.textureReference with
         Address => Texref'Address,
         Import;
      Temp_local_5 : aliased driver_types_h.cudaArray_const_t with
         Address => C_Array'Address,
         Import;
      Temp_call_6 : aliased driver_types_h.cudaChannelFormatDesc with
         Address => Desc'Address,
         Import;

   begin
      declare

         Temp_res_1 : Integer :=
           Integer
             (cuda_runtime_api_h.cudaBindTextureToArray
                (Temp_call_2'Unchecked_Access, Temp_local_5,
                 Temp_call_6'Unchecked_Access));

      begin
         null;

         declare
         begin
            if Temp_res_1 /= 0 then
               Ada.Exceptions.Raise_Exception
                 (CUDA.Exception_Registry.Element (Integer (Temp_res_1)));
            end if;
         end;
      end;
   end Bind_Texture_To_Array;

   procedure Bind_Texture_To_Mipmapped_Array
     (Texref          : out CUDA.Texture_Types.Texture_Reference;
      Mipmapped_Array :     CUDA.Driver_Types.Mipmapped_Array_Const_T;
      Desc            : out CUDA.Driver_Types.Channel_Format_Desc)
   is

      Temp_call_2 : aliased texture_types_h.textureReference with
         Address => Texref'Address,
         Import;
      Temp_local_5 : aliased driver_types_h.cudaMipmappedArray_const_t with
         Address => Mipmapped_Array'Address,
         Import;
      Temp_call_6 : aliased driver_types_h.cudaChannelFormatDesc with
         Address => Desc'Address,
         Import;

   begin
      declare

         Temp_res_1 : Integer :=
           Integer
             (cuda_runtime_api_h.cudaBindTextureToMipmappedArray
                (Temp_call_2'Unchecked_Access, Temp_local_5,
                 Temp_call_6'Unchecked_Access));

      begin
         null;

         declare
         begin
            if Temp_res_1 /= 0 then
               Ada.Exceptions.Raise_Exception
                 (CUDA.Exception_Registry.Element (Integer (Temp_res_1)));
            end if;
         end;
      end;
   end Bind_Texture_To_Mipmapped_Array;

   procedure Unbind_Texture (Texref : out CUDA.Texture_Types.Texture_Reference)
   is

      Temp_call_2 : aliased texture_types_h.textureReference with
         Address => Texref'Address,
         Import;

   begin
      declare

         Temp_res_1 : Integer :=
           Integer
             (cuda_runtime_api_h.cudaUnbindTexture
                (Temp_call_2'Unchecked_Access));

      begin
         null;

         declare
         begin
            if Temp_res_1 /= 0 then
               Ada.Exceptions.Raise_Exception
                 (CUDA.Exception_Registry.Element (Integer (Temp_res_1)));
            end if;
         end;
      end;
   end Unbind_Texture;

   function Get_Texture_Alignment_Offset
     (Texref : out CUDA.Texture_Types.Texture_Reference)
      return CUDA.Corecrt.Size_T
   is

      Temp_call_1 : aliased corecrt_h.size_t;
      Temp_ret_2  : aliased CUDA.Corecrt.Size_T with
         Address => Temp_call_1'Address,
         Import;
      Temp_call_3 : aliased texture_types_h.textureReference with
         Address => Texref'Address,
         Import;

   begin
      declare

         Temp_res_1 : Integer :=
           Integer
             (cuda_runtime_api_h.cudaGetTextureAlignmentOffset
                (Temp_call_1'Unchecked_Access, Temp_call_3'Unchecked_Access));

      begin
         null;

         declare
         begin
            if Temp_res_1 /= 0 then
               Ada.Exceptions.Raise_Exception
                 (CUDA.Exception_Registry.Element (Integer (Temp_res_1)));
            end if;
            return Temp_ret_2;
         end;
      end;
   end Get_Texture_Alignment_Offset;

   procedure Get_Texture_Reference
     (Texref : System.Address; Symbol : System.Address)
   is

      Temp_local_3 : aliased System.Address with
         Address => Texref'Address,
         Import;
      Temp_local_5 : aliased System.Address with
         Address => Symbol'Address,
         Import;

   begin
      declare

         Temp_res_1 : Integer :=
           Integer
             (cuda_runtime_api_h.cudaGetTextureReference
                (Temp_local_3, Temp_local_5));

      begin
         null;

         declare
         begin
            if Temp_res_1 /= 0 then
               Ada.Exceptions.Raise_Exception
                 (CUDA.Exception_Registry.Element (Integer (Temp_res_1)));
            end if;
         end;
      end;
   end Get_Texture_Reference;

   procedure Bind_Surface_To_Array
     (Surfref : out CUDA.Surface_Types.Surface_Reference;
      C_Array :     CUDA.Driver_Types.CUDA_Array_const_t;
      Desc    : out CUDA.Driver_Types.Channel_Format_Desc)
   is

      Temp_call_2 : aliased surface_types_h.surfaceReference with
         Address => Surfref'Address,
         Import;
      Temp_local_5 : aliased driver_types_h.cudaArray_const_t with
         Address => C_Array'Address,
         Import;
      Temp_call_6 : aliased driver_types_h.cudaChannelFormatDesc with
         Address => Desc'Address,
         Import;

   begin
      declare

         Temp_res_1 : Integer :=
           Integer
             (cuda_runtime_api_h.cudaBindSurfaceToArray
                (Temp_call_2'Unchecked_Access, Temp_local_5,
                 Temp_call_6'Unchecked_Access));

      begin
         null;

         declare
         begin
            if Temp_res_1 /= 0 then
               Ada.Exceptions.Raise_Exception
                 (CUDA.Exception_Registry.Element (Integer (Temp_res_1)));
            end if;
         end;
      end;
   end Bind_Surface_To_Array;

   procedure Get_Surface_Reference
     (Surfref : System.Address; Symbol : System.Address)
   is

      Temp_local_3 : aliased System.Address with
         Address => Surfref'Address,
         Import;
      Temp_local_5 : aliased System.Address with
         Address => Symbol'Address,
         Import;

   begin
      declare

         Temp_res_1 : Integer :=
           Integer
             (cuda_runtime_api_h.cudaGetSurfaceReference
                (Temp_local_3, Temp_local_5));

      begin
         null;

         declare
         begin
            if Temp_res_1 /= 0 then
               Ada.Exceptions.Raise_Exception
                 (CUDA.Exception_Registry.Element (Integer (Temp_res_1)));
            end if;
         end;
      end;
   end Get_Surface_Reference;

   function Get_Channel_Desc
     (C_Array : CUDA.Driver_Types.CUDA_Array_const_t)
      return CUDA.Driver_Types.Channel_Format_Desc
   is

      Temp_call_1 : aliased driver_types_h.cudaChannelFormatDesc;
      Temp_ret_2  : aliased CUDA.Driver_Types.Channel_Format_Desc with
         Address => Temp_call_1'Address,
         Import;
      Temp_local_4 : aliased driver_types_h.cudaArray_const_t with
         Address => C_Array'Address,
         Import;

   begin
      declare

         Temp_res_1 : Integer :=
           Integer
             (cuda_runtime_api_h.cudaGetChannelDesc
                (Temp_call_1'Unchecked_Access, Temp_local_4));

      begin
         null;

         declare
         begin
            if Temp_res_1 /= 0 then
               Ada.Exceptions.Raise_Exception
                 (CUDA.Exception_Registry.Element (Integer (Temp_res_1)));
            end if;
            return Temp_ret_2;
         end;
      end;
   end Get_Channel_Desc;

   function Create_Channel_Desc
     (X : int; Y : int; Z : int; W : int;
      F : CUDA.Driver_Types.Channel_Format_Kind)
      return CUDA.Driver_Types.Channel_Format_Desc
   is

      Temp_local_2 : aliased int with
         Address => X'Address,
         Import;
      Temp_local_4 : aliased int with
         Address => Y'Address,
         Import;
      Temp_local_6 : aliased int with
         Address => Z'Address,
         Import;
      Temp_local_8 : aliased int with
         Address => W'Address,
         Import;
      Temp_local_10 : aliased driver_types_h.cudaChannelFormatKind with
         Address => F'Address,
         Import;

   begin
      declare

         Temp_result_orig_1 : aliased driver_types_h.cudaChannelFormatDesc :=
           cuda_runtime_api_h.cudaCreateChannelDesc
             (Temp_local_2, Temp_local_4, Temp_local_6, Temp_local_8,
              Temp_local_10);

      begin
         null;

         declare

            Temp_result_wrapped_2 : aliased CUDA.Driver_Types
              .Channel_Format_Desc with
               Address => Temp_result_orig_1'Address,
               Import;

         begin
            return Temp_result_wrapped_2;
         end;
      end;
   end Create_Channel_Desc;

   procedure Create_Texture_Object
     (P_Tex_Object    : out CUDA.Texture_Types.Texture_Object_T;
      P_Res_Desc      : out CUDA.Driver_Types.Resource_Desc;
      P_Tex_Desc      : out CUDA.Texture_Types.Texture_Desc;
      P_Res_View_Desc : out CUDA.Driver_Types.Resource_View_Desc)
   is

      Temp_call_2 : aliased texture_types_h.cudaTextureObject_t with
         Address => P_Tex_Object'Address,
         Import;
      Temp_call_4 : aliased driver_types_h.cudaResourceDesc with
         Address => P_Res_Desc'Address,
         Import;
      Temp_call_6 : aliased texture_types_h.cudaTextureDesc with
         Address => P_Tex_Desc'Address,
         Import;
      Temp_call_8 : aliased driver_types_h.cudaResourceViewDesc with
         Address => P_Res_View_Desc'Address,
         Import;

   begin
      declare

         Temp_res_1 : Integer :=
           Integer
             (cuda_runtime_api_h.cudaCreateTextureObject
                (Temp_call_2'Unchecked_Access, Temp_call_4'Unchecked_Access,
                 Temp_call_6'Unchecked_Access, Temp_call_8'Unchecked_Access));

      begin
         null;

         declare
         begin
            if Temp_res_1 /= 0 then
               Ada.Exceptions.Raise_Exception
                 (CUDA.Exception_Registry.Element (Integer (Temp_res_1)));
            end if;
         end;
      end;
   end Create_Texture_Object;

   procedure Destroy_Texture_Object
     (Tex_Object : CUDA.Texture_Types.Texture_Object_T)
   is

      Temp_local_3 : aliased texture_types_h.cudaTextureObject_t with
         Address => Tex_Object'Address,
         Import;

   begin
      declare

         Temp_res_1 : Integer :=
           Integer (cuda_runtime_api_h.cudaDestroyTextureObject (Temp_local_3));

      begin
         null;

         declare
         begin
            if Temp_res_1 /= 0 then
               Ada.Exceptions.Raise_Exception
                 (CUDA.Exception_Registry.Element (Integer (Temp_res_1)));
            end if;
         end;
      end;
   end Destroy_Texture_Object;

   function Get_Texture_Object_Resource_Desc
     (Tex_Object : CUDA.Texture_Types.Texture_Object_T)
      return CUDA.Driver_Types.Resource_Desc
   is

      Temp_call_1 : aliased driver_types_h.cudaResourceDesc;
      Temp_ret_2  : aliased CUDA.Driver_Types.Resource_Desc with
         Address => Temp_call_1'Address,
         Import;
      Temp_local_4 : aliased texture_types_h.cudaTextureObject_t with
         Address => Tex_Object'Address,
         Import;

   begin
      declare

         Temp_res_1 : Integer :=
           Integer
             (cuda_runtime_api_h.cudaGetTextureObjectResourceDesc
                (Temp_call_1'Unchecked_Access, Temp_local_4));

      begin
         null;

         declare
         begin
            if Temp_res_1 /= 0 then
               Ada.Exceptions.Raise_Exception
                 (CUDA.Exception_Registry.Element (Integer (Temp_res_1)));
            end if;
            return Temp_ret_2;
         end;
      end;
   end Get_Texture_Object_Resource_Desc;

   function Get_Texture_Object_Texture_Desc
     (Tex_Object : CUDA.Texture_Types.Texture_Object_T)
      return CUDA.Texture_Types.Texture_Desc
   is

      Temp_call_1 : aliased texture_types_h.cudaTextureDesc;
      Temp_ret_2  : aliased CUDA.Texture_Types.Texture_Desc with
         Address => Temp_call_1'Address,
         Import;
      Temp_local_4 : aliased texture_types_h.cudaTextureObject_t with
         Address => Tex_Object'Address,
         Import;

   begin
      declare

         Temp_res_1 : Integer :=
           Integer
             (cuda_runtime_api_h.cudaGetTextureObjectTextureDesc
                (Temp_call_1'Unchecked_Access, Temp_local_4));

      begin
         null;

         declare
         begin
            if Temp_res_1 /= 0 then
               Ada.Exceptions.Raise_Exception
                 (CUDA.Exception_Registry.Element (Integer (Temp_res_1)));
            end if;
            return Temp_ret_2;
         end;
      end;
   end Get_Texture_Object_Texture_Desc;

   function Get_Texture_Object_Resource_View_Desc
     (Tex_Object : CUDA.Texture_Types.Texture_Object_T)
      return CUDA.Driver_Types.Resource_View_Desc
   is

      Temp_call_1 : aliased driver_types_h.cudaResourceViewDesc;
      Temp_ret_2  : aliased CUDA.Driver_Types.Resource_View_Desc with
         Address => Temp_call_1'Address,
         Import;
      Temp_local_4 : aliased texture_types_h.cudaTextureObject_t with
         Address => Tex_Object'Address,
         Import;

   begin
      declare

         Temp_res_1 : Integer :=
           Integer
             (cuda_runtime_api_h.cudaGetTextureObjectResourceViewDesc
                (Temp_call_1'Unchecked_Access, Temp_local_4));

      begin
         null;

         declare
         begin
            if Temp_res_1 /= 0 then
               Ada.Exceptions.Raise_Exception
                 (CUDA.Exception_Registry.Element (Integer (Temp_res_1)));
            end if;
            return Temp_ret_2;
         end;
      end;
   end Get_Texture_Object_Resource_View_Desc;

   procedure Create_Surface_Object
     (P_Surf_Object : out CUDA.Surface_Types.Surface_Object_T;
      P_Res_Desc    : out CUDA.Driver_Types.Resource_Desc)
   is

      Temp_call_2 : aliased surface_types_h.cudaSurfaceObject_t with
         Address => P_Surf_Object'Address,
         Import;
      Temp_call_4 : aliased driver_types_h.cudaResourceDesc with
         Address => P_Res_Desc'Address,
         Import;

   begin
      declare

         Temp_res_1 : Integer :=
           Integer
             (cuda_runtime_api_h.cudaCreateSurfaceObject
                (Temp_call_2'Unchecked_Access, Temp_call_4'Unchecked_Access));

      begin
         null;

         declare
         begin
            if Temp_res_1 /= 0 then
               Ada.Exceptions.Raise_Exception
                 (CUDA.Exception_Registry.Element (Integer (Temp_res_1)));
            end if;
         end;
      end;
   end Create_Surface_Object;

   procedure Destroy_Surface_Object
     (Surf_Object : CUDA.Surface_Types.Surface_Object_T)
   is

      Temp_local_3 : aliased surface_types_h.cudaSurfaceObject_t with
         Address => Surf_Object'Address,
         Import;

   begin
      declare

         Temp_res_1 : Integer :=
           Integer (cuda_runtime_api_h.cudaDestroySurfaceObject (Temp_local_3));

      begin
         null;

         declare
         begin
            if Temp_res_1 /= 0 then
               Ada.Exceptions.Raise_Exception
                 (CUDA.Exception_Registry.Element (Integer (Temp_res_1)));
            end if;
         end;
      end;
   end Destroy_Surface_Object;

   function Get_Surface_Object_Resource_Desc
     (Surf_Object : CUDA.Surface_Types.Surface_Object_T)
      return CUDA.Driver_Types.Resource_Desc
   is

      Temp_call_1 : aliased driver_types_h.cudaResourceDesc;
      Temp_ret_2  : aliased CUDA.Driver_Types.Resource_Desc with
         Address => Temp_call_1'Address,
         Import;
      Temp_local_4 : aliased surface_types_h.cudaSurfaceObject_t with
         Address => Surf_Object'Address,
         Import;

   begin
      declare

         Temp_res_1 : Integer :=
           Integer
             (cuda_runtime_api_h.cudaGetSurfaceObjectResourceDesc
                (Temp_call_1'Unchecked_Access, Temp_local_4));

      begin
         null;

         declare
         begin
            if Temp_res_1 /= 0 then
               Ada.Exceptions.Raise_Exception
                 (CUDA.Exception_Registry.Element (Integer (Temp_res_1)));
            end if;
            return Temp_ret_2;
         end;
      end;
   end Get_Surface_Object_Resource_Desc;

   function Driver_Get_Version return int is

      Temp_call_1 : aliased int;
      Temp_ret_2  : aliased int with
         Address => Temp_call_1'Address,
         Import;

   begin
      declare

         Temp_res_1 : Integer :=
           Integer
             (cuda_runtime_api_h.cudaDriverGetVersion
                (Temp_call_1'Unchecked_Access));

      begin
         null;

         declare
         begin
            if Temp_res_1 /= 0 then
               Ada.Exceptions.Raise_Exception
                 (CUDA.Exception_Registry.Element (Integer (Temp_res_1)));
            end if;
            return Temp_ret_2;
         end;
      end;
   end Driver_Get_Version;

   function Runtime_Get_Version return int is

      Temp_call_1 : aliased int;
      Temp_ret_2  : aliased int with
         Address => Temp_call_1'Address,
         Import;

   begin
      declare

         Temp_res_1 : Integer :=
           Integer
             (cuda_runtime_api_h.cudaRuntimeGetVersion
                (Temp_call_1'Unchecked_Access));

      begin
         null;

         declare
         begin
            if Temp_res_1 /= 0 then
               Ada.Exceptions.Raise_Exception
                 (CUDA.Exception_Registry.Element (Integer (Temp_res_1)));
            end if;
            return Temp_ret_2;
         end;
      end;
   end Runtime_Get_Version;

   procedure Graph_Create (P_Graph : System.Address; Flags : unsigned) is

      Temp_local_3 : aliased System.Address with
         Address => P_Graph'Address,
         Import;
      Temp_local_5 : aliased unsigned with
         Address => Flags'Address,
         Import;

   begin
      declare

         Temp_res_1 : Integer :=
           Integer
             (cuda_runtime_api_h.cudaGraphCreate (Temp_local_3, Temp_local_5));

      begin
         null;

         declare
         begin
            if Temp_res_1 /= 0 then
               Ada.Exceptions.Raise_Exception
                 (CUDA.Exception_Registry.Element (Integer (Temp_res_1)));
            end if;
         end;
      end;
   end Graph_Create;

   procedure Graph_Add_Kernel_Node
     (P_Graph_Node   :     System.Address; Graph : CUDA.Driver_Types.Graph_T;
      P_Dependencies : System.Address; Num_Dependencies : CUDA.Corecrt.Size_T;
      P_Node_Params  : out CUDA.Driver_Types.Kernel_Node_Params)
   is

      Temp_local_3 : aliased System.Address with
         Address => P_Graph_Node'Address,
         Import;
      Temp_local_5 : aliased driver_types_h.cudaGraph_t with
         Address => Graph'Address,
         Import;
      Temp_local_7 : aliased System.Address with
         Address => P_Dependencies'Address,
         Import;
      Temp_local_9 : aliased corecrt_h.size_t with
         Address => Num_Dependencies'Address,
         Import;
      Temp_call_10 : aliased driver_types_h.cudaKernelNodeParams with
         Address => P_Node_Params'Address,
         Import;

   begin
      declare

         Temp_res_1 : Integer :=
           Integer
             (cuda_runtime_api_h.cudaGraphAddKernelNode
                (Temp_local_3, Temp_local_5, Temp_local_7, Temp_local_9,
                 Temp_call_10'Unchecked_Access));

      begin
         null;

         declare
         begin
            if Temp_res_1 /= 0 then
               Ada.Exceptions.Raise_Exception
                 (CUDA.Exception_Registry.Element (Integer (Temp_res_1)));
            end if;
         end;
      end;
   end Graph_Add_Kernel_Node;

   procedure Graph_Kernel_Node_Get_Params
     (Node          :     CUDA.Driver_Types.Graph_Node_T;
      P_Node_Params : out CUDA.Driver_Types.Kernel_Node_Params)
   is

      Temp_local_3 : aliased driver_types_h.cudaGraphNode_t with
         Address => Node'Address,
         Import;
      Temp_call_4 : aliased driver_types_h.cudaKernelNodeParams with
         Address => P_Node_Params'Address,
         Import;

   begin
      declare

         Temp_res_1 : Integer :=
           Integer
             (cuda_runtime_api_h.cudaGraphKernelNodeGetParams
                (Temp_local_3, Temp_call_4'Unchecked_Access));

      begin
         null;

         declare
         begin
            if Temp_res_1 /= 0 then
               Ada.Exceptions.Raise_Exception
                 (CUDA.Exception_Registry.Element (Integer (Temp_res_1)));
            end if;
         end;
      end;
   end Graph_Kernel_Node_Get_Params;

   procedure Graph_Kernel_Node_Set_Params
     (Node          :     CUDA.Driver_Types.Graph_Node_T;
      P_Node_Params : out CUDA.Driver_Types.Kernel_Node_Params)
   is

      Temp_local_3 : aliased driver_types_h.cudaGraphNode_t with
         Address => Node'Address,
         Import;
      Temp_call_4 : aliased driver_types_h.cudaKernelNodeParams with
         Address => P_Node_Params'Address,
         Import;

   begin
      declare

         Temp_res_1 : Integer :=
           Integer
             (cuda_runtime_api_h.cudaGraphKernelNodeSetParams
                (Temp_local_3, Temp_call_4'Unchecked_Access));

      begin
         null;

         declare
         begin
            if Temp_res_1 /= 0 then
               Ada.Exceptions.Raise_Exception
                 (CUDA.Exception_Registry.Element (Integer (Temp_res_1)));
            end if;
         end;
      end;
   end Graph_Kernel_Node_Set_Params;

   procedure Graph_Add_Memcpy_Node
     (P_Graph_Node   :     System.Address; Graph : CUDA.Driver_Types.Graph_T;
      P_Dependencies : System.Address; Num_Dependencies : CUDA.Corecrt.Size_T;
      P_Copy_Params  : out CUDA.Driver_Types.Memcpy3_DParms)
   is

      Temp_local_3 : aliased System.Address with
         Address => P_Graph_Node'Address,
         Import;
      Temp_local_5 : aliased driver_types_h.cudaGraph_t with
         Address => Graph'Address,
         Import;
      Temp_local_7 : aliased System.Address with
         Address => P_Dependencies'Address,
         Import;
      Temp_local_9 : aliased corecrt_h.size_t with
         Address => Num_Dependencies'Address,
         Import;
      Temp_call_10 : aliased driver_types_h.cudaMemcpy3DParms with
         Address => P_Copy_Params'Address,
         Import;

   begin
      declare

         Temp_res_1 : Integer :=
           Integer
             (cuda_runtime_api_h.cudaGraphAddMemcpyNode
                (Temp_local_3, Temp_local_5, Temp_local_7, Temp_local_9,
                 Temp_call_10'Unchecked_Access));

      begin
         null;

         declare
         begin
            if Temp_res_1 /= 0 then
               Ada.Exceptions.Raise_Exception
                 (CUDA.Exception_Registry.Element (Integer (Temp_res_1)));
            end if;
         end;
      end;
   end Graph_Add_Memcpy_Node;

   procedure Graph_Memcpy_Node_Get_Params
     (Node          :     CUDA.Driver_Types.Graph_Node_T;
      P_Node_Params : out CUDA.Driver_Types.Memcpy3_DParms)
   is

      Temp_local_3 : aliased driver_types_h.cudaGraphNode_t with
         Address => Node'Address,
         Import;
      Temp_call_4 : aliased driver_types_h.cudaMemcpy3DParms with
         Address => P_Node_Params'Address,
         Import;

   begin
      declare

         Temp_res_1 : Integer :=
           Integer
             (cuda_runtime_api_h.cudaGraphMemcpyNodeGetParams
                (Temp_local_3, Temp_call_4'Unchecked_Access));

      begin
         null;

         declare
         begin
            if Temp_res_1 /= 0 then
               Ada.Exceptions.Raise_Exception
                 (CUDA.Exception_Registry.Element (Integer (Temp_res_1)));
            end if;
         end;
      end;
   end Graph_Memcpy_Node_Get_Params;

   procedure Graph_Memcpy_Node_Set_Params
     (Node          :     CUDA.Driver_Types.Graph_Node_T;
      P_Node_Params : out CUDA.Driver_Types.Memcpy3_DParms)
   is

      Temp_local_3 : aliased driver_types_h.cudaGraphNode_t with
         Address => Node'Address,
         Import;
      Temp_call_4 : aliased driver_types_h.cudaMemcpy3DParms with
         Address => P_Node_Params'Address,
         Import;

   begin
      declare

         Temp_res_1 : Integer :=
           Integer
             (cuda_runtime_api_h.cudaGraphMemcpyNodeSetParams
                (Temp_local_3, Temp_call_4'Unchecked_Access));

      begin
         null;

         declare
         begin
            if Temp_res_1 /= 0 then
               Ada.Exceptions.Raise_Exception
                 (CUDA.Exception_Registry.Element (Integer (Temp_res_1)));
            end if;
         end;
      end;
   end Graph_Memcpy_Node_Set_Params;

   procedure Graph_Add_Memset_Node
     (P_Graph_Node    :     System.Address; Graph : CUDA.Driver_Types.Graph_T;
      P_Dependencies  : System.Address; Num_Dependencies : CUDA.Corecrt.Size_T;
      P_Memset_Params : out CUDA.Driver_Types.Memset_Params)
   is

      Temp_local_3 : aliased System.Address with
         Address => P_Graph_Node'Address,
         Import;
      Temp_local_5 : aliased driver_types_h.cudaGraph_t with
         Address => Graph'Address,
         Import;
      Temp_local_7 : aliased System.Address with
         Address => P_Dependencies'Address,
         Import;
      Temp_local_9 : aliased corecrt_h.size_t with
         Address => Num_Dependencies'Address,
         Import;
      Temp_call_10 : aliased driver_types_h.cudaMemsetParams with
         Address => P_Memset_Params'Address,
         Import;

   begin
      declare

         Temp_res_1 : Integer :=
           Integer
             (cuda_runtime_api_h.cudaGraphAddMemsetNode
                (Temp_local_3, Temp_local_5, Temp_local_7, Temp_local_9,
                 Temp_call_10'Unchecked_Access));

      begin
         null;

         declare
         begin
            if Temp_res_1 /= 0 then
               Ada.Exceptions.Raise_Exception
                 (CUDA.Exception_Registry.Element (Integer (Temp_res_1)));
            end if;
         end;
      end;
   end Graph_Add_Memset_Node;

   procedure Graph_Memset_Node_Get_Params
     (Node          :     CUDA.Driver_Types.Graph_Node_T;
      P_Node_Params : out CUDA.Driver_Types.Memset_Params)
   is

      Temp_local_3 : aliased driver_types_h.cudaGraphNode_t with
         Address => Node'Address,
         Import;
      Temp_call_4 : aliased driver_types_h.cudaMemsetParams with
         Address => P_Node_Params'Address,
         Import;

   begin
      declare

         Temp_res_1 : Integer :=
           Integer
             (cuda_runtime_api_h.cudaGraphMemsetNodeGetParams
                (Temp_local_3, Temp_call_4'Unchecked_Access));

      begin
         null;

         declare
         begin
            if Temp_res_1 /= 0 then
               Ada.Exceptions.Raise_Exception
                 (CUDA.Exception_Registry.Element (Integer (Temp_res_1)));
            end if;
         end;
      end;
   end Graph_Memset_Node_Get_Params;

   procedure Graph_Memset_Node_Set_Params
     (Node          :     CUDA.Driver_Types.Graph_Node_T;
      P_Node_Params : out CUDA.Driver_Types.Memset_Params)
   is

      Temp_local_3 : aliased driver_types_h.cudaGraphNode_t with
         Address => Node'Address,
         Import;
      Temp_call_4 : aliased driver_types_h.cudaMemsetParams with
         Address => P_Node_Params'Address,
         Import;

   begin
      declare

         Temp_res_1 : Integer :=
           Integer
             (cuda_runtime_api_h.cudaGraphMemsetNodeSetParams
                (Temp_local_3, Temp_call_4'Unchecked_Access));

      begin
         null;

         declare
         begin
            if Temp_res_1 /= 0 then
               Ada.Exceptions.Raise_Exception
                 (CUDA.Exception_Registry.Element (Integer (Temp_res_1)));
            end if;
         end;
      end;
   end Graph_Memset_Node_Set_Params;

   procedure Graph_Add_Host_Node
     (P_Graph_Node   :     System.Address; Graph : CUDA.Driver_Types.Graph_T;
      P_Dependencies : System.Address; Num_Dependencies : CUDA.Corecrt.Size_T;
      P_Node_Params  : out CUDA.Driver_Types.Host_Node_Params)
   is

      Temp_local_3 : aliased System.Address with
         Address => P_Graph_Node'Address,
         Import;
      Temp_local_5 : aliased driver_types_h.cudaGraph_t with
         Address => Graph'Address,
         Import;
      Temp_local_7 : aliased System.Address with
         Address => P_Dependencies'Address,
         Import;
      Temp_local_9 : aliased corecrt_h.size_t with
         Address => Num_Dependencies'Address,
         Import;
      Temp_call_10 : aliased driver_types_h.cudaHostNodeParams with
         Address => P_Node_Params'Address,
         Import;

   begin
      declare

         Temp_res_1 : Integer :=
           Integer
             (cuda_runtime_api_h.cudaGraphAddHostNode
                (Temp_local_3, Temp_local_5, Temp_local_7, Temp_local_9,
                 Temp_call_10'Unchecked_Access));

      begin
         null;

         declare
         begin
            if Temp_res_1 /= 0 then
               Ada.Exceptions.Raise_Exception
                 (CUDA.Exception_Registry.Element (Integer (Temp_res_1)));
            end if;
         end;
      end;
   end Graph_Add_Host_Node;

   procedure Graph_Host_Node_Get_Params
     (Node          :     CUDA.Driver_Types.Graph_Node_T;
      P_Node_Params : out CUDA.Driver_Types.Host_Node_Params)
   is

      Temp_local_3 : aliased driver_types_h.cudaGraphNode_t with
         Address => Node'Address,
         Import;
      Temp_call_4 : aliased driver_types_h.cudaHostNodeParams with
         Address => P_Node_Params'Address,
         Import;

   begin
      declare

         Temp_res_1 : Integer :=
           Integer
             (cuda_runtime_api_h.cudaGraphHostNodeGetParams
                (Temp_local_3, Temp_call_4'Unchecked_Access));

      begin
         null;

         declare
         begin
            if Temp_res_1 /= 0 then
               Ada.Exceptions.Raise_Exception
                 (CUDA.Exception_Registry.Element (Integer (Temp_res_1)));
            end if;
         end;
      end;
   end Graph_Host_Node_Get_Params;

   procedure Graph_Host_Node_Set_Params
     (Node          :     CUDA.Driver_Types.Graph_Node_T;
      P_Node_Params : out CUDA.Driver_Types.Host_Node_Params)
   is

      Temp_local_3 : aliased driver_types_h.cudaGraphNode_t with
         Address => Node'Address,
         Import;
      Temp_call_4 : aliased driver_types_h.cudaHostNodeParams with
         Address => P_Node_Params'Address,
         Import;

   begin
      declare

         Temp_res_1 : Integer :=
           Integer
             (cuda_runtime_api_h.cudaGraphHostNodeSetParams
                (Temp_local_3, Temp_call_4'Unchecked_Access));

      begin
         null;

         declare
         begin
            if Temp_res_1 /= 0 then
               Ada.Exceptions.Raise_Exception
                 (CUDA.Exception_Registry.Element (Integer (Temp_res_1)));
            end if;
         end;
      end;
   end Graph_Host_Node_Set_Params;

   procedure Graph_Add_Child_Graph_Node
     (P_Graph_Node   : System.Address; Graph : CUDA.Driver_Types.Graph_T;
      P_Dependencies : System.Address; Num_Dependencies : CUDA.Corecrt.Size_T;
      Child_Graph    : CUDA.Driver_Types.Graph_T)
   is

      Temp_local_3 : aliased System.Address with
         Address => P_Graph_Node'Address,
         Import;
      Temp_local_5 : aliased driver_types_h.cudaGraph_t with
         Address => Graph'Address,
         Import;
      Temp_local_7 : aliased System.Address with
         Address => P_Dependencies'Address,
         Import;
      Temp_local_9 : aliased corecrt_h.size_t with
         Address => Num_Dependencies'Address,
         Import;
      Temp_local_11 : aliased driver_types_h.cudaGraph_t with
         Address => Child_Graph'Address,
         Import;

   begin
      declare

         Temp_res_1 : Integer :=
           Integer
             (cuda_runtime_api_h.cudaGraphAddChildGraphNode
                (Temp_local_3, Temp_local_5, Temp_local_7, Temp_local_9,
                 Temp_local_11));

      begin
         null;

         declare
         begin
            if Temp_res_1 /= 0 then
               Ada.Exceptions.Raise_Exception
                 (CUDA.Exception_Registry.Element (Integer (Temp_res_1)));
            end if;
         end;
      end;
   end Graph_Add_Child_Graph_Node;

   procedure Graph_Child_Graph_Node_Get_Graph
     (Node : CUDA.Driver_Types.Graph_Node_T; P_Graph : System.Address)
   is

      Temp_local_3 : aliased driver_types_h.cudaGraphNode_t with
         Address => Node'Address,
         Import;
      Temp_local_5 : aliased System.Address with
         Address => P_Graph'Address,
         Import;

   begin
      declare

         Temp_res_1 : Integer :=
           Integer
             (cuda_runtime_api_h.cudaGraphChildGraphNodeGetGraph
                (Temp_local_3, Temp_local_5));

      begin
         null;

         declare
         begin
            if Temp_res_1 /= 0 then
               Ada.Exceptions.Raise_Exception
                 (CUDA.Exception_Registry.Element (Integer (Temp_res_1)));
            end if;
         end;
      end;
   end Graph_Child_Graph_Node_Get_Graph;

   procedure Graph_Add_Empty_Node
     (P_Graph_Node   : System.Address; Graph : CUDA.Driver_Types.Graph_T;
      P_Dependencies : System.Address; Num_Dependencies : CUDA.Corecrt.Size_T)
   is

      Temp_local_3 : aliased System.Address with
         Address => P_Graph_Node'Address,
         Import;
      Temp_local_5 : aliased driver_types_h.cudaGraph_t with
         Address => Graph'Address,
         Import;
      Temp_local_7 : aliased System.Address with
         Address => P_Dependencies'Address,
         Import;
      Temp_local_9 : aliased corecrt_h.size_t with
         Address => Num_Dependencies'Address,
         Import;

   begin
      declare

         Temp_res_1 : Integer :=
           Integer
             (cuda_runtime_api_h.cudaGraphAddEmptyNode
                (Temp_local_3, Temp_local_5, Temp_local_7, Temp_local_9));

      begin
         null;

         declare
         begin
            if Temp_res_1 /= 0 then
               Ada.Exceptions.Raise_Exception
                 (CUDA.Exception_Registry.Element (Integer (Temp_res_1)));
            end if;
         end;
      end;
   end Graph_Add_Empty_Node;

   procedure Graph_Clone
     (P_Graph_Clone  : System.Address;
      Original_Graph : CUDA.Driver_Types.Graph_T)
   is

      Temp_local_3 : aliased System.Address with
         Address => P_Graph_Clone'Address,
         Import;
      Temp_local_5 : aliased driver_types_h.cudaGraph_t with
         Address => Original_Graph'Address,
         Import;

   begin
      declare

         Temp_res_1 : Integer :=
           Integer
             (cuda_runtime_api_h.cudaGraphClone (Temp_local_3, Temp_local_5));

      begin
         null;

         declare
         begin
            if Temp_res_1 /= 0 then
               Ada.Exceptions.Raise_Exception
                 (CUDA.Exception_Registry.Element (Integer (Temp_res_1)));
            end if;
         end;
      end;
   end Graph_Clone;

   procedure Graph_Node_Find_In_Clone
     (P_Node : System.Address; Original_Node : CUDA.Driver_Types.Graph_Node_T;
      Cloned_Graph : CUDA.Driver_Types.Graph_T)
   is

      Temp_local_3 : aliased System.Address with
         Address => P_Node'Address,
         Import;
      Temp_local_5 : aliased driver_types_h.cudaGraphNode_t with
         Address => Original_Node'Address,
         Import;
      Temp_local_7 : aliased driver_types_h.cudaGraph_t with
         Address => Cloned_Graph'Address,
         Import;

   begin
      declare

         Temp_res_1 : Integer :=
           Integer
             (cuda_runtime_api_h.cudaGraphNodeFindInClone
                (Temp_local_3, Temp_local_5, Temp_local_7));

      begin
         null;

         declare
         begin
            if Temp_res_1 /= 0 then
               Ada.Exceptions.Raise_Exception
                 (CUDA.Exception_Registry.Element (Integer (Temp_res_1)));
            end if;
         end;
      end;
   end Graph_Node_Find_In_Clone;

   procedure Graph_Node_Get_Type
     (Node   :     CUDA.Driver_Types.Graph_Node_T;
      P_Type : out CUDA.Driver_Types.Graph_Node_Type)
   is

      Temp_local_3 : aliased driver_types_h.cudaGraphNode_t with
         Address => Node'Address,
         Import;
      Temp_call_4 : aliased driver_types_h.cudaGraphNodeType with
         Address => P_Type'Address,
         Import;

   begin
      declare

         Temp_res_1 : Integer :=
           Integer
             (cuda_runtime_api_h.cudaGraphNodeGetType
                (Temp_local_3, Temp_call_4'Unchecked_Access));

      begin
         null;

         declare
         begin
            if Temp_res_1 /= 0 then
               Ada.Exceptions.Raise_Exception
                 (CUDA.Exception_Registry.Element (Integer (Temp_res_1)));
            end if;
         end;
      end;
   end Graph_Node_Get_Type;

   procedure Graph_Get_Nodes
     (Graph     :     CUDA.Driver_Types.Graph_T; Nodes : System.Address;
      Num_Nodes : out CUDA.Corecrt.Size_T)
   is

      Temp_local_3 : aliased driver_types_h.cudaGraph_t with
         Address => Graph'Address,
         Import;
      Temp_local_5 : aliased System.Address with
         Address => Nodes'Address,
         Import;
      Temp_call_6 : aliased corecrt_h.size_t with
         Address => Num_Nodes'Address,
         Import;

   begin
      declare

         Temp_res_1 : Integer :=
           Integer
             (cuda_runtime_api_h.cudaGraphGetNodes
                (Temp_local_3, Temp_local_5, Temp_call_6'Unchecked_Access));

      begin
         null;

         declare
         begin
            if Temp_res_1 /= 0 then
               Ada.Exceptions.Raise_Exception
                 (CUDA.Exception_Registry.Element (Integer (Temp_res_1)));
            end if;
         end;
      end;
   end Graph_Get_Nodes;

   procedure Graph_Get_Root_Nodes
     (Graph :     CUDA.Driver_Types.Graph_T; P_Root_Nodes : System.Address;
      P_Num_Root_Nodes : out CUDA.Corecrt.Size_T)
   is

      Temp_local_3 : aliased driver_types_h.cudaGraph_t with
         Address => Graph'Address,
         Import;
      Temp_local_5 : aliased System.Address with
         Address => P_Root_Nodes'Address,
         Import;
      Temp_call_6 : aliased corecrt_h.size_t with
         Address => P_Num_Root_Nodes'Address,
         Import;

   begin
      declare

         Temp_res_1 : Integer :=
           Integer
             (cuda_runtime_api_h.cudaGraphGetRootNodes
                (Temp_local_3, Temp_local_5, Temp_call_6'Unchecked_Access));

      begin
         null;

         declare
         begin
            if Temp_res_1 /= 0 then
               Ada.Exceptions.Raise_Exception
                 (CUDA.Exception_Registry.Element (Integer (Temp_res_1)));
            end if;
         end;
      end;
   end Graph_Get_Root_Nodes;

   procedure Graph_Get_Edges
     (Graph : CUDA.Driver_Types.Graph_T; From : System.Address;
      To    : System.Address; Num_Edges : out CUDA.Corecrt.Size_T)
   is

      Temp_local_3 : aliased driver_types_h.cudaGraph_t with
         Address => Graph'Address,
         Import;
      Temp_local_5 : aliased System.Address with
         Address => From'Address,
         Import;
      Temp_local_7 : aliased System.Address with
         Address => To'Address,
         Import;
      Temp_call_8 : aliased corecrt_h.size_t with
         Address => Num_Edges'Address,
         Import;

   begin
      declare

         Temp_res_1 : Integer :=
           Integer
             (cuda_runtime_api_h.cudaGraphGetEdges
                (Temp_local_3, Temp_local_5, Temp_local_7,
                 Temp_call_8'Unchecked_Access));

      begin
         null;

         declare
         begin
            if Temp_res_1 /= 0 then
               Ada.Exceptions.Raise_Exception
                 (CUDA.Exception_Registry.Element (Integer (Temp_res_1)));
            end if;
         end;
      end;
   end Graph_Get_Edges;

   procedure Graph_Node_Get_Dependencies
     (Node : CUDA.Driver_Types.Graph_Node_T; P_Dependencies : System.Address;
      P_Num_Dependencies : out CUDA.Corecrt.Size_T)
   is

      Temp_local_3 : aliased driver_types_h.cudaGraphNode_t with
         Address => Node'Address,
         Import;
      Temp_local_5 : aliased System.Address with
         Address => P_Dependencies'Address,
         Import;
      Temp_call_6 : aliased corecrt_h.size_t with
         Address => P_Num_Dependencies'Address,
         Import;

   begin
      declare

         Temp_res_1 : Integer :=
           Integer
             (cuda_runtime_api_h.cudaGraphNodeGetDependencies
                (Temp_local_3, Temp_local_5, Temp_call_6'Unchecked_Access));

      begin
         null;

         declare
         begin
            if Temp_res_1 /= 0 then
               Ada.Exceptions.Raise_Exception
                 (CUDA.Exception_Registry.Element (Integer (Temp_res_1)));
            end if;
         end;
      end;
   end Graph_Node_Get_Dependencies;

   procedure Graph_Node_Get_Dependent_Nodes
     (Node : CUDA.Driver_Types.Graph_Node_T; P_Dependent_Nodes : System.Address;
      P_Num_Dependent_Nodes : out CUDA.Corecrt.Size_T)
   is

      Temp_local_3 : aliased driver_types_h.cudaGraphNode_t with
         Address => Node'Address,
         Import;
      Temp_local_5 : aliased System.Address with
         Address => P_Dependent_Nodes'Address,
         Import;
      Temp_call_6 : aliased corecrt_h.size_t with
         Address => P_Num_Dependent_Nodes'Address,
         Import;

   begin
      declare

         Temp_res_1 : Integer :=
           Integer
             (cuda_runtime_api_h.cudaGraphNodeGetDependentNodes
                (Temp_local_3, Temp_local_5, Temp_call_6'Unchecked_Access));

      begin
         null;

         declare
         begin
            if Temp_res_1 /= 0 then
               Ada.Exceptions.Raise_Exception
                 (CUDA.Exception_Registry.Element (Integer (Temp_res_1)));
            end if;
         end;
      end;
   end Graph_Node_Get_Dependent_Nodes;

   procedure Graph_Add_Dependencies
     (Graph : CUDA.Driver_Types.Graph_T; From : System.Address;
      To    : System.Address; Num_Dependencies : CUDA.Corecrt.Size_T)
   is

      Temp_local_3 : aliased driver_types_h.cudaGraph_t with
         Address => Graph'Address,
         Import;
      Temp_local_5 : aliased System.Address with
         Address => From'Address,
         Import;
      Temp_local_7 : aliased System.Address with
         Address => To'Address,
         Import;
      Temp_local_9 : aliased corecrt_h.size_t with
         Address => Num_Dependencies'Address,
         Import;

   begin
      declare

         Temp_res_1 : Integer :=
           Integer
             (cuda_runtime_api_h.cudaGraphAddDependencies
                (Temp_local_3, Temp_local_5, Temp_local_7, Temp_local_9));

      begin
         null;

         declare
         begin
            if Temp_res_1 /= 0 then
               Ada.Exceptions.Raise_Exception
                 (CUDA.Exception_Registry.Element (Integer (Temp_res_1)));
            end if;
         end;
      end;
   end Graph_Add_Dependencies;

   procedure Graph_Remove_Dependencies
     (Graph : CUDA.Driver_Types.Graph_T; From : System.Address;
      To    : System.Address; Num_Dependencies : CUDA.Corecrt.Size_T)
   is

      Temp_local_3 : aliased driver_types_h.cudaGraph_t with
         Address => Graph'Address,
         Import;
      Temp_local_5 : aliased System.Address with
         Address => From'Address,
         Import;
      Temp_local_7 : aliased System.Address with
         Address => To'Address,
         Import;
      Temp_local_9 : aliased corecrt_h.size_t with
         Address => Num_Dependencies'Address,
         Import;

   begin
      declare

         Temp_res_1 : Integer :=
           Integer
             (cuda_runtime_api_h.cudaGraphRemoveDependencies
                (Temp_local_3, Temp_local_5, Temp_local_7, Temp_local_9));

      begin
         null;

         declare
         begin
            if Temp_res_1 /= 0 then
               Ada.Exceptions.Raise_Exception
                 (CUDA.Exception_Registry.Element (Integer (Temp_res_1)));
            end if;
         end;
      end;
   end Graph_Remove_Dependencies;

   procedure Graph_Destroy_Node (Node : CUDA.Driver_Types.Graph_Node_T) is

      Temp_local_3 : aliased driver_types_h.cudaGraphNode_t with
         Address => Node'Address,
         Import;

   begin
      declare

         Temp_res_1 : Integer :=
           Integer (cuda_runtime_api_h.cudaGraphDestroyNode (Temp_local_3));

      begin
         null;

         declare
         begin
            if Temp_res_1 /= 0 then
               Ada.Exceptions.Raise_Exception
                 (CUDA.Exception_Registry.Element (Integer (Temp_res_1)));
            end if;
         end;
      end;
   end Graph_Destroy_Node;

   procedure Graph_Instantiate
     (P_Graph_Exec : System.Address; Graph : CUDA.Driver_Types.Graph_T;
      P_Error_Node : System.Address; P_Log_Buffer : String;
      Buffer_Size  : CUDA.Corecrt.Size_T)
   is

      Temp_local_3 : aliased System.Address with
         Address => P_Graph_Exec'Address,
         Import;
      Temp_local_5 : aliased driver_types_h.cudaGraph_t with
         Address => Graph'Address,
         Import;
      Temp_local_7 : aliased System.Address with
         Address => P_Error_Node'Address,
         Import;
      Temp_c_string_8 : Interfaces.C.Strings.chars_ptr :=
        Interfaces.C.Strings.New_String (P_Log_Buffer);
      Temp_local_11 : aliased corecrt_h.size_t with
         Address => Buffer_Size'Address,
         Import;

   begin
      declare

         Temp_res_1 : Integer :=
           Integer
             (cuda_runtime_api_h.cudaGraphInstantiate
                (Temp_local_3, Temp_local_5, Temp_local_7, Temp_c_string_8,
                 Temp_local_11));

      begin
         null;

         declare
         begin
            if Temp_res_1 /= 0 then
               Ada.Exceptions.Raise_Exception
                 (CUDA.Exception_Registry.Element (Integer (Temp_res_1)));
            end if;
            Interfaces.C.Strings.Free (Temp_c_string_8);
         end;
      end;
   end Graph_Instantiate;

   procedure Graph_Exec_Kernel_Node_Set_Params
     (H_Graph_Exec  :     CUDA.Driver_Types.Graph_Exec_T;
      Node          :     CUDA.Driver_Types.Graph_Node_T;
      P_Node_Params : out CUDA.Driver_Types.Kernel_Node_Params)
   is

      Temp_local_3 : aliased driver_types_h.cudaGraphExec_t with
         Address => H_Graph_Exec'Address,
         Import;
      Temp_local_5 : aliased driver_types_h.cudaGraphNode_t with
         Address => Node'Address,
         Import;
      Temp_call_6 : aliased driver_types_h.cudaKernelNodeParams with
         Address => P_Node_Params'Address,
         Import;

   begin
      declare

         Temp_res_1 : Integer :=
           Integer
             (cuda_runtime_api_h.cudaGraphExecKernelNodeSetParams
                (Temp_local_3, Temp_local_5, Temp_call_6'Unchecked_Access));

      begin
         null;

         declare
         begin
            if Temp_res_1 /= 0 then
               Ada.Exceptions.Raise_Exception
                 (CUDA.Exception_Registry.Element (Integer (Temp_res_1)));
            end if;
         end;
      end;
   end Graph_Exec_Kernel_Node_Set_Params;

   procedure Graph_Exec_Memcpy_Node_Set_Params
     (H_Graph_Exec  :     CUDA.Driver_Types.Graph_Exec_T;
      Node          :     CUDA.Driver_Types.Graph_Node_T;
      P_Node_Params : out CUDA.Driver_Types.Memcpy3_DParms)
   is

      Temp_local_3 : aliased driver_types_h.cudaGraphExec_t with
         Address => H_Graph_Exec'Address,
         Import;
      Temp_local_5 : aliased driver_types_h.cudaGraphNode_t with
         Address => Node'Address,
         Import;
      Temp_call_6 : aliased driver_types_h.cudaMemcpy3DParms with
         Address => P_Node_Params'Address,
         Import;

   begin
      declare

         Temp_res_1 : Integer :=
           Integer
             (cuda_runtime_api_h.cudaGraphExecMemcpyNodeSetParams
                (Temp_local_3, Temp_local_5, Temp_call_6'Unchecked_Access));

      begin
         null;

         declare
         begin
            if Temp_res_1 /= 0 then
               Ada.Exceptions.Raise_Exception
                 (CUDA.Exception_Registry.Element (Integer (Temp_res_1)));
            end if;
         end;
      end;
   end Graph_Exec_Memcpy_Node_Set_Params;

   procedure Graph_Exec_Memset_Node_Set_Params
     (H_Graph_Exec  :     CUDA.Driver_Types.Graph_Exec_T;
      Node          :     CUDA.Driver_Types.Graph_Node_T;
      P_Node_Params : out CUDA.Driver_Types.Memset_Params)
   is

      Temp_local_3 : aliased driver_types_h.cudaGraphExec_t with
         Address => H_Graph_Exec'Address,
         Import;
      Temp_local_5 : aliased driver_types_h.cudaGraphNode_t with
         Address => Node'Address,
         Import;
      Temp_call_6 : aliased driver_types_h.cudaMemsetParams with
         Address => P_Node_Params'Address,
         Import;

   begin
      declare

         Temp_res_1 : Integer :=
           Integer
             (cuda_runtime_api_h.cudaGraphExecMemsetNodeSetParams
                (Temp_local_3, Temp_local_5, Temp_call_6'Unchecked_Access));

      begin
         null;

         declare
         begin
            if Temp_res_1 /= 0 then
               Ada.Exceptions.Raise_Exception
                 (CUDA.Exception_Registry.Element (Integer (Temp_res_1)));
            end if;
         end;
      end;
   end Graph_Exec_Memset_Node_Set_Params;

   procedure Graph_Exec_Host_Node_Set_Params
     (H_Graph_Exec  :     CUDA.Driver_Types.Graph_Exec_T;
      Node          :     CUDA.Driver_Types.Graph_Node_T;
      P_Node_Params : out CUDA.Driver_Types.Host_Node_Params)
   is

      Temp_local_3 : aliased driver_types_h.cudaGraphExec_t with
         Address => H_Graph_Exec'Address,
         Import;
      Temp_local_5 : aliased driver_types_h.cudaGraphNode_t with
         Address => Node'Address,
         Import;
      Temp_call_6 : aliased driver_types_h.cudaHostNodeParams with
         Address => P_Node_Params'Address,
         Import;

   begin
      declare

         Temp_res_1 : Integer :=
           Integer
             (cuda_runtime_api_h.cudaGraphExecHostNodeSetParams
                (Temp_local_3, Temp_local_5, Temp_call_6'Unchecked_Access));

      begin
         null;

         declare
         begin
            if Temp_res_1 /= 0 then
               Ada.Exceptions.Raise_Exception
                 (CUDA.Exception_Registry.Element (Integer (Temp_res_1)));
            end if;
         end;
      end;
   end Graph_Exec_Host_Node_Set_Params;

   procedure Graph_Exec_Update
     (H_Graph_Exec      :     CUDA.Driver_Types.Graph_Exec_T;
      H_Graph : CUDA.Driver_Types.Graph_T; H_Error_Node_Out : System.Address;
      Update_Result_Out : out CUDA.Driver_Types.Graph_Exec_Update_Result)
   is

      Temp_local_3 : aliased driver_types_h.cudaGraphExec_t with
         Address => H_Graph_Exec'Address,
         Import;
      Temp_local_5 : aliased driver_types_h.cudaGraph_t with
         Address => H_Graph'Address,
         Import;
      Temp_local_7 : aliased System.Address with
         Address => H_Error_Node_Out'Address,
         Import;
      Temp_call_8 : aliased driver_types_h.cudaGraphExecUpdateResult with
         Address => Update_Result_Out'Address,
         Import;

   begin
      declare

         Temp_res_1 : Integer :=
           Integer
             (cuda_runtime_api_h.cudaGraphExecUpdate
                (Temp_local_3, Temp_local_5, Temp_local_7,
                 Temp_call_8'Unchecked_Access));

      begin
         null;

         declare
         begin
            if Temp_res_1 /= 0 then
               Ada.Exceptions.Raise_Exception
                 (CUDA.Exception_Registry.Element (Integer (Temp_res_1)));
            end if;
         end;
      end;
   end Graph_Exec_Update;

   procedure Graph_Launch
     (Graph_Exec : CUDA.Driver_Types.Graph_Exec_T;
      Stream     : CUDA.Driver_Types.Stream_T)
   is

      Temp_local_3 : aliased driver_types_h.cudaGraphExec_t with
         Address => Graph_Exec'Address,
         Import;
      Temp_local_5 : aliased driver_types_h.cudaStream_t with
         Address => Stream'Address,
         Import;

   begin
      declare

         Temp_res_1 : Integer :=
           Integer
             (cuda_runtime_api_h.cudaGraphLaunch (Temp_local_3, Temp_local_5));

      begin
         null;

         declare
         begin
            if Temp_res_1 /= 0 then
               Ada.Exceptions.Raise_Exception
                 (CUDA.Exception_Registry.Element (Integer (Temp_res_1)));
            end if;
         end;
      end;
   end Graph_Launch;

   procedure Graph_Exec_Destroy (Graph_Exec : CUDA.Driver_Types.Graph_Exec_T) is

      Temp_local_3 : aliased driver_types_h.cudaGraphExec_t with
         Address => Graph_Exec'Address,
         Import;

   begin
      declare

         Temp_res_1 : Integer :=
           Integer (cuda_runtime_api_h.cudaGraphExecDestroy (Temp_local_3));

      begin
         null;

         declare
         begin
            if Temp_res_1 /= 0 then
               Ada.Exceptions.Raise_Exception
                 (CUDA.Exception_Registry.Element (Integer (Temp_res_1)));
            end if;
         end;
      end;
   end Graph_Exec_Destroy;

   procedure Graph_Destroy (Graph : CUDA.Driver_Types.Graph_T) is

      Temp_local_3 : aliased driver_types_h.cudaGraph_t with
         Address => Graph'Address,
         Import;

   begin
      declare

         Temp_res_1 : Integer :=
           Integer (cuda_runtime_api_h.cudaGraphDestroy (Temp_local_3));

      begin
         null;

         declare
         begin
            if Temp_res_1 /= 0 then
               Ada.Exceptions.Raise_Exception
                 (CUDA.Exception_Registry.Element (Integer (Temp_res_1)));
            end if;
         end;
      end;
   end Graph_Destroy;

   procedure Get_Export_Table
     (Pp_Export_Table   :     System.Address;
      P_Export_Table_Id : out CUDA.Driver_Types.UUID_T)
   is

      Temp_local_3 : aliased System.Address with
         Address => Pp_Export_Table'Address,
         Import;
      Temp_call_4 : aliased driver_types_h.cudaUUID_t with
         Address => P_Export_Table_Id'Address,
         Import;

   begin
      declare

         Temp_res_1 : Integer :=
           Integer
             (cuda_runtime_api_h.cudaGetExportTable
                (Temp_local_3, Temp_call_4'Unchecked_Access));

      begin
         null;

         declare
         begin
            if Temp_res_1 /= 0 then
               Ada.Exceptions.Raise_Exception
                 (CUDA.Exception_Registry.Element (Integer (Temp_res_1)));
            end if;
         end;
      end;
   end Get_Export_Table;

begin
   null;
end CUDA.Runtime_Api;
