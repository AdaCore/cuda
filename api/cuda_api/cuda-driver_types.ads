with Interfaces.C; use Interfaces.C;
with System;
with CUDA.Stddef;
with stddef_h;
with Interfaces.C.Extensions;
with CUDA.Vector_Types;
with uvector_types_h;

package CUDA.Driver_Types is

   subtype Error is unsigned;
   Success : unsigned;
   type Channel_Format_Kind is
     (Channel_Format_Kind_Signed, Channel_Format_Kind_Unsigned,
      Channel_Format_Kind_Float, Channel_Format_Kind_None) with
      Convention => C;

   type Channel_Format_Desc is record
      X : int;
      Y : int;
      Z : int;
      W : int;
      F : Channel_Format_Kind;

   end record with
      Convention => C;

   type CUDA_Array is null record;

   type CUDA_Array_t is access CUDA_Array;

   type CUDA_Array_const_t is access CUDA_Array;

   type Mipmapped_Array is null record;

   type Mipmapped_Array_T is access Mipmapped_Array;

   type Mipmapped_Array_Const_T is access Mipmapped_Array;

   type Memory_Type_T is
     (Memory_Type_Unregistered, Memory_Type_Host, Memory_Type_Device,
      Memory_Type_Managed) with
      Convention => C;

   type Memcpy_Kind is
     (Memcpy_Host_To_Host, Memcpy_Host_To_Device, Memcpy_Device_To_Host,
      Memcpy_Device_To_Device, Memcpy_Default) with
      Convention => C;

   type Pitched_Ptr is record
      Ptr   : System.Address;
      Pitch : CUDA.Stddef.Size_T;
      Xsize : CUDA.Stddef.Size_T;
      Ysize : CUDA.Stddef.Size_T;

   end record with
      Convention => C;

   type Extent_T is record
      Width  : CUDA.Stddef.Size_T;
      Height : CUDA.Stddef.Size_T;
      Depth  : CUDA.Stddef.Size_T;

   end record with
      Convention => C;

   type Pos is record
      X : CUDA.Stddef.Size_T;
      Y : CUDA.Stddef.Size_T;
      Z : CUDA.Stddef.Size_T;

   end record with
      Convention => C;

   type Memcpy3_DParms is record
      Src_Array : CUDA_Array_t;
      Src_Pos   : Pos;
      Src_Ptr   : Pitched_Ptr;
      Dst_Array : CUDA_Array_t;
      Dst_Pos   : Pos;
      Dst_Ptr   : Pitched_Ptr;
      Extent    : Extent_T;
      Kind      : Memcpy_Kind;

   end record with
      Convention => C;

   type Memcpy3_DPeer_Parms is record
      Src_Array  : CUDA_Array_t;
      Src_Pos    : Pos;
      Src_Ptr    : Pitched_Ptr;
      Src_Device : int;
      Dst_Array  : CUDA_Array_t;
      Dst_Pos    : Pos;
      Dst_Ptr    : Pitched_Ptr;
      Dst_Device : int;
      Extent     : Extent_T;

   end record with
      Convention => C;

   type Memset_Params is record
      Dst          : System.Address;
      Pitch        : CUDA.Stddef.Size_T;
      Value        : unsigned;
      Element_Size : unsigned;
      Width        : CUDA.Stddef.Size_T;
      Height       : CUDA.Stddef.Size_T;

   end record with
      Convention => C;

   type Access_Property is
     (Access_Property_Normal, Access_Property_Streaming,
      Access_Property_Persisting) with
      Convention => C;

   type Access_Policy_Window_T is record
      Base_Ptr  : System.Address;
      Num_Bytes : CUDA.Stddef.Size_T;
      Hit_Ratio : Float;
      Hit_Prop  : Access_Property;
      Miss_Prop : Access_Property;

   end record with
      Convention => C;

   type Host_Fn_T is access procedure (arg1 : System.Address) with
      Convention => C;

   generic
      with procedure Temp_Call_1 (Arg1 : System.Address);
   procedure Host_Fn_T_Gen (Arg1 : System.Address);
   type Host_Node_Params is record
      Fn        : Host_Fn_T;
      User_Data : System.Address;

   end record with
      Convention => C;

   type Stream_Capture_Status is
     (Stream_Capture_Status_None, Stream_Capture_Status_Active,
      Stream_Capture_Status_Invalidated) with
      Convention => C;

   type Stream_Capture_Mode is
     (Stream_Capture_Mode_Global, Stream_Capture_Mode_Thread_Local,
      Stream_Capture_Mode_Relaxed) with
      Convention => C;

   subtype Synchronization_Policy is unsigned;
   Sync_Policy_Auto          : unsigned;
   Sync_Policy_Spin          : unsigned;
   Sync_Policy_Yield         : unsigned;
   Sync_Policy_Blocking_Sync : unsigned;
   subtype Stream_Attr_ID is unsigned;
   Stream_Attribute_Access_Policy_Window   : unsigned;
   Stream_Attribute_Synchronization_Policy : unsigned;
   type Stream_Attr_Value is record
      Access_Policy_Window : Access_Policy_Window_T;
      Sync_Policy          : Synchronization_Policy;

   end record with
      Convention => C;

   type Graphics_Resource is null record;

   subtype Graphics_Register_Flags is unsigned;
   Graphics_Register_Flags_None               : unsigned;
   Graphics_Register_Flags_Read_Only          : unsigned;
   Graphics_Register_Flags_Write_Discard      : unsigned;
   Graphics_Register_Flags_Surface_Load_Store : unsigned;
   Graphics_Register_Flags_Texture_Gather     : unsigned;
   type Graphics_Map_Flags is
     (Graphics_Map_Flags_None, Graphics_Map_Flags_Read_Only,
      Graphics_Map_Flags_Write_Discard) with
      Convention => C;

   type Graphics_Cube_Face is
     (Graphics_Cube_Face_Positive_X, Graphics_Cube_Face_Negative_X,
      Graphics_Cube_Face_Positive_Y, Graphics_Cube_Face_Negative_Y,
      Graphics_Cube_Face_Positive_Z, Graphics_Cube_Face_Negative_Z) with
      Convention => C;

   subtype Kernel_Node_Attr_ID is unsigned;
   Kernel_Node_Attribute_Access_Policy_Window : unsigned;
   Kernel_Node_Attribute_Cooperative          : unsigned;
   type Kernel_Node_Attr_Value is record
      Access_Policy_Window : Access_Policy_Window_T;
      Cooperative          : int;

   end record with
      Convention => C;

   type Resource_Type is
     (Resource_Type_Array, Resource_Type_Mipmapped_Array, Resource_Type_Linear,
      Resource_Type_Pitch2_D) with
      Convention => C;

   type Resource_View_Format is
     (Res_View_Format_None, Res_View_Format_Unsigned_Char1,
      Res_View_Format_Unsigned_Char2, Res_View_Format_Unsigned_Char4,
      Res_View_Format_Signed_Char1, Res_View_Format_Signed_Char2,
      Res_View_Format_Signed_Char4, Res_View_Format_Unsigned_Short1,
      Res_View_Format_Unsigned_Short2, Res_View_Format_Unsigned_Short4,
      Res_View_Format_Signed_Short1, Res_View_Format_Signed_Short2,
      Res_View_Format_Signed_Short4, Res_View_Format_Unsigned_Int1,
      Res_View_Format_Unsigned_Int2, Res_View_Format_Unsigned_Int4,
      Res_View_Format_Signed_Int1, Res_View_Format_Signed_Int2,
      Res_View_Format_Signed_Int4, Res_View_Format_Half1, Res_View_Format_Half2,
      Res_View_Format_Half4, Res_View_Format_Float1, Res_View_Format_Float2,
      Res_View_Format_Float4, Res_View_Format_Unsigned_Block_Compressed1,
      Res_View_Format_Unsigned_Block_Compressed2,
      Res_View_Format_Unsigned_Block_Compressed3,
      Res_View_Format_Unsigned_Block_Compressed4,
      Res_View_Format_Signed_Block_Compressed4,
      Res_View_Format_Unsigned_Block_Compressed5,
      Res_View_Format_Signed_Block_Compressed5,
      Res_View_Format_Unsigned_Block_Compressed6_H,
      Res_View_Format_Signed_Block_Compressed6_H,
      Res_View_Format_Unsigned_Block_Compressed7) with
      Convention => C;

   type Anon936_Struct938 is record
      C_Array : CUDA_Array_t;

   end record with
      Convention => C;

   type Anon936_Struct939 is record
      Mipmap : Mipmapped_Array_T;

   end record with
      Convention => C;

   type Anon936_Struct940 is record
      Dev_Ptr       : System.Address;
      Desc          : Channel_Format_Desc;
      Size_In_Bytes : CUDA.Stddef.Size_T;

   end record with
      Convention => C;

   type Anon936_Struct941 is record
      Dev_Ptr        : System.Address;
      Desc           : Channel_Format_Desc;
      Width          : CUDA.Stddef.Size_T;
      Height         : CUDA.Stddef.Size_T;
      Pitch_In_Bytes : CUDA.Stddef.Size_T;

   end record with
      Convention => C;

   type Anon936_Union937 is record
      C_Array  : Anon936_Struct938;
      Mipmap   : Anon936_Struct939;
      Linear   : Anon936_Struct940;
      Pitch2_D : Anon936_Struct941;

   end record with
      Convention => C;

   type Resource_Desc is record
      Res_Type : Resource_Type;
      Res      : Anon936_Union937;

   end record with
      Convention => C;

   type Resource_View_Desc is record
      Format             : Resource_View_Format;
      Width              : CUDA.Stddef.Size_T;
      Height             : CUDA.Stddef.Size_T;
      Depth              : CUDA.Stddef.Size_T;
      First_Mipmap_Level : unsigned;
      Last_Mipmap_Level  : unsigned;
      First_Layer        : unsigned;
      Last_Layer         : unsigned;

   end record with
      Convention => C;

   type Pointer_Attributes is record
      C_Type         : Memory_Type_T;
      Device         : int;
      Device_Pointer : System.Address;
      Host_Pointer   : System.Address;

   end record with
      Convention => C;

   type Func_Attributes is record
      Shared_Size_Bytes             : CUDA.Stddef.Size_T;
      Const_Size_Bytes              : CUDA.Stddef.Size_T;
      Local_Size_Bytes              : CUDA.Stddef.Size_T;
      Max_Threads_Per_Block         : int;
      Num_Regs                      : int;
      Ptx_Version                   : int;
      Binary_Version                : int;
      Cache_Mode_CA                 : int;
      Max_Dynamic_Shared_Size_Bytes : int;
      Preferred_Shmem_Carveout      : int;

   end record with
      Convention => C;

   subtype Func_Attribute is unsigned;
   Func_Attribute_Max_Dynamic_Shared_Memory_Size   : unsigned;
   Func_Attribute_Preferred_Shared_Memory_Carveout : unsigned;
   Func_Attribute_Max                              : unsigned;
   type Func_Cache is
     (Func_Cache_Prefer_None, Func_Cache_Prefer_Shared, Func_Cache_Prefer_L1,
      Func_Cache_Prefer_Equal) with
      Convention => C;

   type Shared_Mem_Config is
     (Shared_Mem_Bank_Size_Default, Shared_Mem_Bank_Size_Four_Byte,
      Shared_Mem_Bank_Size_Eight_Byte) with
      Convention => C;

   subtype Shared_Carveout is int;
   Sharedmem_Carveout_Default    : int;
   Sharedmem_Carveout_Max_Shared : int;
   Sharedmem_Carveout_Max_L1     : int;
   type Compute_Mode is
     (Compute_Mode_Default, Compute_Mode_Exclusive, Compute_Mode_Prohibited,
      Compute_Mode_Exclusive_Process) with
      Convention => C;

   type Limit is
     (Limit_Stack_Size, Limit_Printf_Fifo_Size, Limit_Malloc_Heap_Size,
      Limit_Dev_Runtime_Sync_Depth, Limit_Dev_Runtime_Pending_Launch_Count,
      Limit_Max_L2_Fetch_Granularity, Limit_Persisting_L2_Cache_Size) with
      Convention => C;

   subtype Memory_Advise is unsigned;
   Mem_Advise_Set_Read_Mostly          : unsigned;
   Mem_Advise_Unset_Read_Mostly        : unsigned;
   Mem_Advise_Set_Preferred_Location   : unsigned;
   Mem_Advise_Unset_Preferred_Location : unsigned;
   Mem_Advise_Set_Accessed_By          : unsigned;
   Mem_Advise_Unset_Accessed_By        : unsigned;
   subtype Mem_Range_Attribute is unsigned;
   Mem_Range_Attribute_Read_Mostly            : unsigned;
   Mem_Range_Attribute_Preferred_Location     : unsigned;
   Mem_Range_Attribute_Accessed_By            : unsigned;
   Mem_Range_Attribute_Last_Prefetch_Location : unsigned;
   type Output_Mode is (Key_Value_Pair, CSV) with
      Convention => C;

   subtype Device_Attr is unsigned;
   Dev_Attr_Max_Threads_Per_Block                        : unsigned;
   Dev_Attr_Max_Block_Dim_X                              : unsigned;
   Dev_Attr_Max_Block_Dim_Y                              : unsigned;
   Dev_Attr_Max_Block_Dim_Z                              : unsigned;
   Dev_Attr_Max_Grid_Dim_X                               : unsigned;
   Dev_Attr_Max_Grid_Dim_Y                               : unsigned;
   Dev_Attr_Max_Grid_Dim_Z                               : unsigned;
   Dev_Attr_Max_Shared_Memory_Per_Block                  : unsigned;
   Dev_Attr_Total_Constant_Memory                        : unsigned;
   Dev_Attr_Warp_Size                                    : unsigned;
   Dev_Attr_Max_Pitch                                    : unsigned;
   Dev_Attr_Max_Registers_Per_Block                      : unsigned;
   Dev_Attr_Clock_Rate                                   : unsigned;
   Dev_Attr_Texture_Alignment                            : unsigned;
   Dev_Attr_Gpu_Overlap                                  : unsigned;
   Dev_Attr_Multi_Processor_Count                        : unsigned;
   Dev_Attr_Kernel_Exec_Timeout                          : unsigned;
   Dev_Attr_Integrated                                   : unsigned;
   Dev_Attr_Can_Map_Host_Memory                          : unsigned;
   Dev_Attr_Compute_Mode                                 : unsigned;
   Dev_Attr_Max_Texture1_DWidth                          : unsigned;
   Dev_Attr_Max_Texture2_DWidth                          : unsigned;
   Dev_Attr_Max_Texture2_DHeight                         : unsigned;
   Dev_Attr_Max_Texture3_DWidth                          : unsigned;
   Dev_Attr_Max_Texture3_DHeight                         : unsigned;
   Dev_Attr_Max_Texture3_DDepth                          : unsigned;
   Dev_Attr_Max_Texture2_DLayered_Width                  : unsigned;
   Dev_Attr_Max_Texture2_DLayered_Height                 : unsigned;
   Dev_Attr_Max_Texture2_DLayered_Layers                 : unsigned;
   Dev_Attr_Surface_Alignment                            : unsigned;
   Dev_Attr_Concurrent_Kernels                           : unsigned;
   Dev_Attr_Ecc_Enabled                                  : unsigned;
   Dev_Attr_Pci_Bus_Id                                   : unsigned;
   Dev_Attr_Pci_Device_Id                                : unsigned;
   Dev_Attr_Tcc_Driver                                   : unsigned;
   Dev_Attr_Memory_Clock_Rate                            : unsigned;
   Dev_Attr_Global_Memory_Bus_Width                      : unsigned;
   Dev_Attr_L2_Cache_Size                                : unsigned;
   Dev_Attr_Max_Threads_Per_Multi_Processor              : unsigned;
   Dev_Attr_Async_Engine_Count                           : unsigned;
   Dev_Attr_Unified_Addressing                           : unsigned;
   Dev_Attr_Max_Texture1_DLayered_Width                  : unsigned;
   Dev_Attr_Max_Texture1_DLayered_Layers                 : unsigned;
   Dev_Attr_Max_Texture2_DGather_Width                   : unsigned;
   Dev_Attr_Max_Texture2_DGather_Height                  : unsigned;
   Dev_Attr_Max_Texture3_DWidth_Alt                      : unsigned;
   Dev_Attr_Max_Texture3_DHeight_Alt                     : unsigned;
   Dev_Attr_Max_Texture3_DDepth_Alt                      : unsigned;
   Dev_Attr_Pci_Domain_Id                                : unsigned;
   Dev_Attr_Texture_Pitch_Alignment                      : unsigned;
   Dev_Attr_Max_Texture_Cubemap_Width                    : unsigned;
   Dev_Attr_Max_Texture_Cubemap_Layered_Width            : unsigned;
   Dev_Attr_Max_Texture_Cubemap_Layered_Layers           : unsigned;
   Dev_Attr_Max_Surface1_DWidth                          : unsigned;
   Dev_Attr_Max_Surface2_DWidth                          : unsigned;
   Dev_Attr_Max_Surface2_DHeight                         : unsigned;
   Dev_Attr_Max_Surface3_DWidth                          : unsigned;
   Dev_Attr_Max_Surface3_DHeight                         : unsigned;
   Dev_Attr_Max_Surface3_DDepth                          : unsigned;
   Dev_Attr_Max_Surface1_DLayered_Width                  : unsigned;
   Dev_Attr_Max_Surface1_DLayered_Layers                 : unsigned;
   Dev_Attr_Max_Surface2_DLayered_Width                  : unsigned;
   Dev_Attr_Max_Surface2_DLayered_Height                 : unsigned;
   Dev_Attr_Max_Surface2_DLayered_Layers                 : unsigned;
   Dev_Attr_Max_Surface_Cubemap_Width                    : unsigned;
   Dev_Attr_Max_Surface_Cubemap_Layered_Width            : unsigned;
   Dev_Attr_Max_Surface_Cubemap_Layered_Layers           : unsigned;
   Dev_Attr_Max_Texture1_DLinear_Width                   : unsigned;
   Dev_Attr_Max_Texture2_DLinear_Width                   : unsigned;
   Dev_Attr_Max_Texture2_DLinear_Height                  : unsigned;
   Dev_Attr_Max_Texture2_DLinear_Pitch                   : unsigned;
   Dev_Attr_Max_Texture2_DMipmapped_Width                : unsigned;
   Dev_Attr_Max_Texture2_DMipmapped_Height               : unsigned;
   Dev_Attr_Compute_Capability_Major                     : unsigned;
   Dev_Attr_Compute_Capability_Minor                     : unsigned;
   Dev_Attr_Max_Texture1_DMipmapped_Width                : unsigned;
   Dev_Attr_Stream_Priorities_Supported                  : unsigned;
   Dev_Attr_Global_L1_Cache_Supported                    : unsigned;
   Dev_Attr_Local_L1_Cache_Supported                     : unsigned;
   Dev_Attr_Max_Shared_Memory_Per_Multiprocessor         : unsigned;
   Dev_Attr_Max_Registers_Per_Multiprocessor             : unsigned;
   Dev_Attr_Managed_Memory                               : unsigned;
   Dev_Attr_Is_Multi_Gpu_Board                           : unsigned;
   Dev_Attr_Multi_Gpu_Board_Group_ID                     : unsigned;
   Dev_Attr_Host_Native_Atomic_Supported                 : unsigned;
   Dev_Attr_Single_To_Double_Precision_Perf_Ratio        : unsigned;
   Dev_Attr_Pageable_Memory_Access                       : unsigned;
   Dev_Attr_Concurrent_Managed_Access                    : unsigned;
   Dev_Attr_Compute_Preemption_Supported                 : unsigned;
   Dev_Attr_Can_Use_Host_Pointer_For_Registered_Mem      : unsigned;
   Dev_Attr_Reserved92                                   : unsigned;
   Dev_Attr_Reserved93                                   : unsigned;
   Dev_Attr_Reserved94                                   : unsigned;
   Dev_Attr_Cooperative_Launch                           : unsigned;
   Dev_Attr_Cooperative_Multi_Device_Launch              : unsigned;
   Dev_Attr_Max_Shared_Memory_Per_Block_Optin            : unsigned;
   Dev_Attr_Can_Flush_Remote_Writes                      : unsigned;
   Dev_Attr_Host_Register_Supported                      : unsigned;
   Dev_Attr_Pageable_Memory_Access_Uses_Host_Page_Tables : unsigned;
   Dev_Attr_Direct_Managed_Mem_Access_From_Host          : unsigned;
   Dev_Attr_Max_Blocks_Per_Multiprocessor                : unsigned;
   Dev_Attr_Reserved_Shared_Memory_Per_Block             : unsigned;
   subtype Device_P2_PAttr is unsigned;
   Dev_P2_PAttr_Performance_Rank            : unsigned;
   Dev_P2_PAttr_Access_Supported            : unsigned;
   Dev_P2_PAttr_Native_Atomic_Supported     : unsigned;
   Dev_P2_PAttr_Cuda_Array_Access_Supported : unsigned;
   subtype Anon956_Array958 is Interfaces.C.char_array (0 .. 15);
   type CUuuid_St is record
      Bytes : Anon956_Array958;

   end record with
      Convention => C;

   subtype CUuuid is CUuuid_St;
   subtype UUID_T is CUuuid_St;
   subtype Anon961_Array963 is Interfaces.C.char_array (0 .. 255);
   subtype Anon961_Array965 is Interfaces.C.char_array (0 .. 7);
   type Anon961_Array967 is array (0 .. 2) of int;

   type Anon961_Array969 is array (0 .. 1) of int;

   type Device_Prop is record
      Name                                         : Anon961_Array963;
      Uuid                                         : UUID_T;
      Luid                                         : Anon961_Array965;
      Luid_Device_Node_Mask                        : unsigned;
      Total_Global_Mem                             : CUDA.Stddef.Size_T;
      Shared_Mem_Per_Block                         : CUDA.Stddef.Size_T;
      Regs_Per_Block                               : int;
      Warp_Size                                    : int;
      Mem_Pitch                                    : CUDA.Stddef.Size_T;
      Max_Threads_Per_Block                        : int;
      Max_Threads_Dim                              : Anon961_Array967;
      Max_Grid_Size                                : Anon961_Array967;
      Clock_Rate                                   : int;
      Total_Const_Mem                              : CUDA.Stddef.Size_T;
      Major                                        : int;
      Minor                                        : int;
      Texture_Alignment                            : CUDA.Stddef.Size_T;
      Texture_Pitch_Alignment                      : CUDA.Stddef.Size_T;
      Device_Overlap                               : int;
      Multi_Processor_Count                        : int;
      Kernel_Exec_Timeout_Enabled                  : int;
      Integrated                                   : int;
      Can_Map_Host_Memory                          : int;
      Compute_Mode                                 : int;
      Max_Texture1_D                               : int;
      Max_Texture1_DMipmap                         : int;
      Max_Texture1_DLinear                         : int;
      Max_Texture2_D                               : Anon961_Array969;
      Max_Texture2_DMipmap                         : Anon961_Array969;
      Max_Texture2_DLinear                         : Anon961_Array967;
      Max_Texture2_DGather                         : Anon961_Array969;
      Max_Texture3_D                               : Anon961_Array967;
      Max_Texture3_DAlt                            : Anon961_Array967;
      Max_Texture_Cubemap                          : int;
      Max_Texture1_DLayered                        : Anon961_Array969;
      Max_Texture2_DLayered                        : Anon961_Array967;
      Max_Texture_Cubemap_Layered                  : Anon961_Array969;
      Max_Surface1_D                               : int;
      Max_Surface2_D                               : Anon961_Array969;
      Max_Surface3_D                               : Anon961_Array967;
      Max_Surface1_DLayered                        : Anon961_Array969;
      Max_Surface2_DLayered                        : Anon961_Array967;
      Max_Surface_Cubemap                          : int;
      Max_Surface_Cubemap_Layered                  : Anon961_Array969;
      Surface_Alignment                            : CUDA.Stddef.Size_T;
      Concurrent_Kernels                           : int;
      ECCEnabled                                   : int;
      Pci_Bus_ID                                   : int;
      Pci_Device_ID                                : int;
      Pci_Domain_ID                                : int;
      Tcc_Driver                                   : int;
      Async_Engine_Count                           : int;
      Unified_Addressing                           : int;
      Memory_Clock_Rate                            : int;
      Memory_Bus_Width                             : int;
      L2_Cache_Size                                : int;
      Persisting_L2_Cache_Max_Size                 : int;
      Max_Threads_Per_Multi_Processor              : int;
      Stream_Priorities_Supported                  : int;
      Global_L1_Cache_Supported                    : int;
      Local_L1_Cache_Supported                     : int;
      Shared_Mem_Per_Multiprocessor                : CUDA.Stddef.Size_T;
      Regs_Per_Multiprocessor                      : int;
      Managed_Memory                               : int;
      Is_Multi_Gpu_Board                           : int;
      Multi_Gpu_Board_Group_ID                     : int;
      Host_Native_Atomic_Supported                 : int;
      Single_To_Double_Precision_Perf_Ratio        : int;
      Pageable_Memory_Access                       : int;
      Concurrent_Managed_Access                    : int;
      Compute_Preemption_Supported                 : int;
      Can_Use_Host_Pointer_For_Registered_Mem      : int;
      Cooperative_Launch                           : int;
      Cooperative_Multi_Device_Launch              : int;
      Shared_Mem_Per_Block_Optin                   : CUDA.Stddef.Size_T;
      Pageable_Memory_Access_Uses_Host_Page_Tables : int;
      Direct_Managed_Mem_Access_From_Host          : int;
      Max_Blocks_Per_Multi_Processor               : int;
      Access_Policy_Max_Window_Size                : int;
      Reserved_Shared_Mem_Per_Block                : CUDA.Stddef.Size_T;

   end record with
      Convention => C;

   subtype Anon970_Array972 is Interfaces.C.char_array (0 .. 63);
   type Ipc_Event_Handle_St is record
      Reserved : Anon970_Array972;

   end record with
      Convention => C;

   subtype Ipc_Event_Handle_T is Ipc_Event_Handle_St;
   subtype Anon974_Array972 is Interfaces.C.char_array (0 .. 63);
   type Ipc_Mem_Handle_St is record
      Reserved : Anon974_Array972;

   end record with
      Convention => C;

   subtype Ipc_Mem_Handle_T is Ipc_Mem_Handle_St;
   subtype External_Memory_Handle_Type is unsigned;
   External_Memory_Handle_Type_Opaque_Fd           : unsigned;
   External_Memory_Handle_Type_Opaque_Win32        : unsigned;
   External_Memory_Handle_Type_Opaque_Win32_Kmt    : unsigned;
   External_Memory_Handle_Type_D3_D12_Heap         : unsigned;
   External_Memory_Handle_Type_D3_D12_Resource     : unsigned;
   External_Memory_Handle_Type_D3_D11_Resource     : unsigned;
   External_Memory_Handle_Type_D3_D11_Resource_Kmt : unsigned;
   External_Memory_Handle_Type_Nv_Sci_Buf          : unsigned;
   type Anon977_Struct979 is record
      Handle : System.Address;
      Name   : System.Address;

   end record with
      Convention => C;

   type Anon977_Union978 is record
      Fd                : int;
      Win32             : Anon977_Struct979;
      Nv_Sci_Buf_Object : System.Address;

   end record with
      Convention => C;

   type External_Memory_Handle_Desc is record
      C_Type : External_Memory_Handle_Type;
      Handle : Anon977_Union978;
      Size   : Extensions.unsigned_long_long;
      Flags  : unsigned;

   end record with
      Convention => C;

   type External_Memory_Buffer_Desc is record
      Offset : Extensions.unsigned_long_long;
      Size   : Extensions.unsigned_long_long;
      Flags  : unsigned;

   end record with
      Convention => C;

   type External_Memory_Mipmapped_Array_Desc is record
      Offset      : Extensions.unsigned_long_long;
      Format_Desc : Channel_Format_Desc;
      Extent      : Extent_T;
      Flags       : unsigned;
      Num_Levels  : unsigned;

   end record with
      Convention => C;

   subtype External_Semaphore_Handle_Type is unsigned;
   External_Semaphore_Handle_Type_Opaque_Fd        : unsigned;
   External_Semaphore_Handle_Type_Opaque_Win32     : unsigned;
   External_Semaphore_Handle_Type_Opaque_Win32_Kmt : unsigned;
   External_Semaphore_Handle_Type_D3_D12_Fence     : unsigned;
   External_Semaphore_Handle_Type_D3_D11_Fence     : unsigned;
   External_Semaphore_Handle_Type_Nv_Sci_Sync      : unsigned;
   External_Semaphore_Handle_Type_Keyed_Mutex      : unsigned;
   External_Semaphore_Handle_Type_Keyed_Mutex_Kmt  : unsigned;
   type Anon983_Struct985 is record
      Handle : System.Address;
      Name   : System.Address;

   end record with
      Convention => C;

   type Anon983_Union984 is record
      Fd              : int;
      Win32           : Anon983_Struct985;
      Nv_Sci_Sync_Obj : System.Address;

   end record with
      Convention => C;

   type External_Semaphore_Handle_Desc is record
      C_Type : External_Semaphore_Handle_Type;
      Handle : Anon983_Union984;
      Flags  : unsigned;

   end record with
      Convention => C;

   type Anon986_Struct988 is record
      Value : Extensions.unsigned_long_long;

   end record with
      Convention => C;

   type Anon986_Union989 is record
      Fence    : System.Address;
      Reserved : Extensions.unsigned_long_long;

   end record with
      Convention => C;

   type Anon986_Struct990 is record
      Key : Extensions.unsigned_long_long;

   end record with
      Convention => C;

   type Anon986_Struct987 is record
      Fence       : Anon986_Struct988;
      Nv_Sci_Sync : Anon986_Union989;
      Keyed_Mutex : Anon986_Struct990;

   end record with
      Convention => C;

   type External_Semaphore_Signal_Params is record
      Params : Anon986_Struct987;
      Flags  : unsigned;

   end record with
      Convention => C;

   type Anon991_Struct993 is record
      Value : Extensions.unsigned_long_long;

   end record with
      Convention => C;

   type Anon991_Union994 is record
      Fence    : System.Address;
      Reserved : Extensions.unsigned_long_long;

   end record with
      Convention => C;

   type Anon991_Struct995 is record
      Key        : Extensions.unsigned_long_long;
      Timeout_Ms : unsigned;

   end record with
      Convention => C;

   type Anon991_Struct992 is record
      Fence       : Anon991_Struct993;
      Nv_Sci_Sync : Anon991_Union994;
      Keyed_Mutex : Anon991_Struct995;

   end record with
      Convention => C;

   type External_Semaphore_Wait_Params is record
      Params : Anon991_Struct992;
      Flags  : unsigned;

   end record with
      Convention => C;

   subtype Error_T is Error;
   type CUstream_St is null record;

   type Stream_T is access CUstream_St;

   type CUevent_St is null record;

   type Event_T is access CUevent_St;

   type Graphics_Resource_T is access Graphics_Resource;

   subtype Output_Mode_T is Output_Mode;
   type CUexternal_Memory_St is null record;

   type External_Memory_T is access CUexternal_Memory_St;

   type CUexternal_Semaphore_St is null record;

   type External_Semaphore_T is access CUexternal_Semaphore_St;

   type CUgraph_St is null record;

   type Graph_T is access CUgraph_St;

   type CUgraph_Node_St is null record;

   type Graph_Node_T is access CUgraph_Node_St;

   type CUfunc_St is null record;

   type Function_T is access CUfunc_St;

   type CGScope is (CGScope_Invalid, CGScope_Grid, CGScope_Multi_Grid) with
      Convention => C;

   type Launch_Params is record
      Func       : System.Address;
      Grid_Dim   : CUDA.Vector_Types.Dim3;
      Block_Dim  : CUDA.Vector_Types.Dim3;
      Args       : System.Address;
      Shared_Mem : CUDA.Stddef.Size_T;
      Stream     : Stream_T;

   end record with
      Convention => C;

   type Kernel_Node_Params is record
      Func             : System.Address;
      Grid_Dim         : CUDA.Vector_Types.Dim3;
      Block_Dim        : CUDA.Vector_Types.Dim3;
      Shared_Mem_Bytes : unsigned;
      Kernel_Params    : System.Address;
      Extra            : System.Address;

   end record with
      Convention => C;

   type Graph_Node_Type is
     (Graph_Node_Type_Kernel, Graph_Node_Type_Memcpy, Graph_Node_Type_Memset,
      Graph_Node_Type_Host, Graph_Node_Type_Graph, Graph_Node_Type_Empty,
      Graph_Node_Type_Count) with
      Convention => C;

   type CUgraph_Exec_St is null record;

   type Graph_Exec_T is access CUgraph_Exec_St;

   type Graph_Exec_Update_Result is
     (Graph_Exec_Update_Success, Graph_Exec_Update_Error,
      Graph_Exec_Update_Error_Topology_Changed,
      Graph_Exec_Update_Error_Node_Type_Changed,
      Graph_Exec_Update_Error_Function_Changed,
      Graph_Exec_Update_Error_Parameters_Changed,
      Graph_Exec_Update_Error_Not_Supported) with
      Convention => C;

end CUDA.Driver_Types;
