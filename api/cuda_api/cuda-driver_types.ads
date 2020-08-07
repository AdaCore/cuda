with Interfaces.C; use Interfaces.C;
with System;
with CUDA.Corecrt;
with CUDA.Vector_Types;
with Interfaces.C.Extensions;

package CUDA.Driver_Types is
   pragma Elaborate_Body;
   Host_Alloc_Default                                 : constant := 16#00#;
   Host_Alloc_Portable                                : constant := 16#01#;
   Host_Alloc_Mapped                                  : constant := 16#02#;
   Host_Alloc_Write_Combined                          : constant := 16#04#;
   Host_Register_Default                              : constant := 16#00#;
   Host_Register_Portable                             : constant := 16#01#;
   Host_Register_Mapped                               : constant := 16#02#;
   Host_Register_Io_Memory                            : constant := 16#04#;
   Peer_Access_Default                                : constant := 16#00#;
   Stream_Default                                     : constant := 16#00#;
   Stream_Non_Blocking                                : constant := 16#01#;
   Event_Default                                      : constant := 16#00#;
   Event_Blocking_Sync                                : constant := 16#01#;
   Event_Disable_Timing                               : constant := 16#02#;
   Event_Interprocess                                 : constant := 16#04#;
   Device_Schedule_Auto                               : constant := 16#00#;
   Device_Schedule_Spin                               : constant := 16#01#;
   Device_Schedule_Yield                              : constant := 16#02#;
   Device_Schedule_Blocking_Sync                      : constant := 16#04#;
   Device_Blocking_Sync                               : constant := 16#04#;
   Device_Schedule_Mask                               : constant := 16#07#;
   Device_Map_Host                                    : constant := 16#08#;
   Device_Lmem_Resize_To_Max                          : constant := 16#10#;
   Device_Mask                                        : constant := 16#1f#;
   CUDA_Array_Default                                 : constant := 16#00#;
   CUDA_Array_Layered                                 : constant := 16#01#;
   CUDA_Array_Surface_Load_Store                      : constant := 16#02#;
   CUDA_Array_Cubemap                                 : constant := 16#04#;
   CUDA_Array_Texture_Gather                          : constant := 16#08#;
   CUDA_Array_Color_Attachment                        : constant := 16#20#;
   Ipc_Mem_Lazy_Enable_Peer_Access                    : constant := 16#01#;
   Mem_Attach_Global                                  : constant := 16#01#;
   Mem_Attach_Host                                    : constant := 16#02#;
   Mem_Attach_Single                                  : constant := 16#04#;
   Occupancy_Default                                  : constant := 16#00#;
   Occupancy_Disable_Caching_Override                 : constant := 16#01#;
   Cooperative_Launch_Multi_Device_No_Pre_Sync        : constant := 16#01#;
   Cooperative_Launch_Multi_Device_No_Post_Sync       : constant := 16#02#;
   CUDA_IPC_HANDLE_SIZE                               : constant := 64;
   External_Memory_Dedicated                          : constant := 16#1#;
   External_Semaphore_Signal_Skip_Nv_Sci_Buf_Mem_Sync : constant := 16#01#;
   External_Semaphore_Wait_Skip_Nv_Sci_Buf_Mem_Sync   : constant := 16#02#;
   Nv_Sci_Sync_Attr_Signal                            : constant := 16#1#;
   Nv_Sci_Sync_Attr_Wait                              : constant := 16#2#;
   subtype Error is Interfaces.C.unsigned;
   Success : constant Interfaces.C.unsigned := 0;
   Error_Invalid_Value                  : exception;
   Error_Memory_Allocation              : exception;
   Error_Initialization_Error           : exception;
   Error_Cudart_Unloading               : exception;
   Error_Profiler_Disabled              : exception;
   Error_Profiler_Not_Initialized       : exception;
   Error_Profiler_Already_Started       : exception;
   Error_Profiler_Already_Stopped       : exception;
   Error_Invalid_Configuration          : exception;
   Error_Invalid_Pitch_Value            : exception;
   Error_Invalid_Symbol                 : exception;
   Error_Invalid_Host_Pointer           : exception;
   Error_Invalid_Device_Pointer         : exception;
   Error_Invalid_Texture                : exception;
   Error_Invalid_Texture_Binding        : exception;
   Error_Invalid_Channel_Descriptor     : exception;
   Error_Invalid_Memcpy_Direction       : exception;
   Error_Address_Of_Constant            : exception;
   Error_Texture_Fetch_Failed           : exception;
   Error_Texture_Not_Bound              : exception;
   Error_Synchronization_Error          : exception;
   Error_Invalid_Filter_Setting         : exception;
   Error_Invalid_Norm_Setting           : exception;
   Error_Mixed_Device_Execution         : exception;
   Error_Not_Yet_Implemented            : exception;
   Error_Memory_Value_Too_Large         : exception;
   Error_Insufficient_Driver            : exception;
   Error_Invalid_Surface                : exception;
   Error_Duplicate_Variable_Name        : exception;
   Error_Duplicate_Texture_Name         : exception;
   Error_Duplicate_Surface_Name         : exception;
   Error_Devices_Unavailable            : exception;
   Error_Incompatible_Driver_Context    : exception;
   Error_Missing_Configuration          : exception;
   Error_Prior_Launch_Failure           : exception;
   Error_Launch_Max_Depth_Exceeded      : exception;
   Error_Launch_File_Scoped_Tex         : exception;
   Error_Launch_File_Scoped_Surf        : exception;
   Error_Sync_Depth_Exceeded            : exception;
   Error_Launch_Pending_Count_Exceeded  : exception;
   Error_Invalid_Device_Function        : exception;
   Error_No_Device                      : exception;
   Error_Invalid_Device                 : exception;
   Error_Startup_Failure                : exception;
   Error_Invalid_Kernel_Image           : exception;
   Error_Device_Uninitialized           : exception;
   Error_Map_Buffer_Object_Failed       : exception;
   Error_Unmap_Buffer_Object_Failed     : exception;
   Error_Array_Is_Mapped                : exception;
   Error_Already_Mapped                 : exception;
   Error_No_Kernel_Image_For_Device     : exception;
   Error_Already_Acquired               : exception;
   Error_Not_Mapped                     : exception;
   Error_Not_Mapped_As_Array            : exception;
   Error_Not_Mapped_As_Pointer          : exception;
   Error_ECCUncorrectable               : exception;
   Error_Unsupported_Limit              : exception;
   Error_Device_Already_In_Use          : exception;
   Error_Peer_Access_Unsupported        : exception;
   Error_Invalid_Ptx                    : exception;
   Error_Invalid_Graphics_Context       : exception;
   Error_Nvlink_Uncorrectable           : exception;
   Error_Jit_Compiler_Not_Found         : exception;
   Error_Invalid_Source                 : exception;
   Error_File_Not_Found                 : exception;
   Error_Shared_Object_Symbol_Not_Found : exception;
   Error_Shared_Object_Init_Failed      : exception;
   Error_Operating_System               : exception;
   Error_Invalid_Resource_Handle        : exception;
   Error_Illegal_State                  : exception;
   Error_Symbol_Not_Found               : exception;
   Error_Not_Ready                      : exception;
   Error_Illegal_Address                : exception;
   Error_Launch_Out_Of_Resources        : exception;
   Error_Launch_Timeout                 : exception;
   Error_Launch_Incompatible_Texturing  : exception;
   Error_Peer_Access_Already_Enabled    : exception;
   Error_Peer_Access_Not_Enabled        : exception;
   Error_Set_On_Active_Process          : exception;
   Error_Context_Is_Destroyed           : exception;
   Error_Assert                         : exception;
   Error_Too_Many_Peers                 : exception;
   Error_Host_Memory_Already_Registered : exception;
   Error_Host_Memory_Not_Registered     : exception;
   Error_Hardware_Stack_Error           : exception;
   Error_Illegal_Instruction            : exception;
   Error_Misaligned_Address             : exception;
   Error_Invalid_Address_Space          : exception;
   Error_Invalid_Pc                     : exception;
   Error_Launch_Failure                 : exception;
   Error_Cooperative_Launch_Too_Large   : exception;
   Error_Not_Permitted                  : exception;
   Error_Not_Supported                  : exception;
   Error_System_Not_Ready               : exception;
   Error_System_Driver_Mismatch         : exception;
   Error_Compat_Not_Supported_On_Device : exception;
   Error_Stream_Capture_Unsupported     : exception;
   Error_Stream_Capture_Invalidated     : exception;
   Error_Stream_Capture_Merge           : exception;
   Error_Stream_Capture_Unmatched       : exception;
   Error_Stream_Capture_Unjoined        : exception;
   Error_Stream_Capture_Isolation       : exception;
   Error_Stream_Capture_Implicit        : exception;
   Error_Captured_Event                 : exception;
   Error_Stream_Capture_Wrong_Thread    : exception;
   Error_Timeout                        : exception;
   Error_Graph_Exec_Update_Failure      : exception;
   Error_Unknown                        : exception;
   Error_Api_Failure_Base               : exception;

   type Channel_Format_Kind is (Channel_Format_Kind_Signed, Channel_Format_Kind_Unsigned, Channel_Format_Kind_Float, Channel_Format_Kind_None) with
      Convention => C;

   type Channel_Format_Desc is record
      X : aliased Interfaces.C.int;
      Y : aliased Interfaces.C.int;
      Z : aliased Interfaces.C.int;
      W : aliased Interfaces.C.int;
      F : aliased Channel_Format_Kind;
   end record with
      Convention => C_Pass_By_Copy;

   type CUDA_Array is null record;

   type CUDA_Array_t is access CUDA_Array;

   type CUDA_Array_const_t is access CUDA_Array;

   type Mipmapped_Array is null record;

   type Mipmapped_Array_T is access Mipmapped_Array;

   type Mipmapped_Array_Const_T is access Mipmapped_Array;

   type Memory_Type_T is
     (Memory_Type_Unregistered, Memory_Type_Host, Memory_Type_Device,
      Memory_Type_Managed);

   type Memcpy_Kind is
     (Memcpy_Host_To_Host, Memcpy_Host_To_Device, Memcpy_Device_To_Host,
      Memcpy_Device_To_Device, Memcpy_Default);

   type Pitched_Ptr is record
      Ptr   : System.Address;
      Pitch : CUDA.Corecrt.Size_T;
      Xsize : CUDA.Corecrt.Size_T;
      Ysize : CUDA.Corecrt.Size_T;
   end record;

   type Extent_T is record
      Width  : CUDA.Corecrt.Size_T;
      Height : CUDA.Corecrt.Size_T;
      Depth  : CUDA.Corecrt.Size_T;
   end record;

   type Pos is record
      X : CUDA.Corecrt.Size_T;
      Y : CUDA.Corecrt.Size_T;
      Z : CUDA.Corecrt.Size_T;
   end record;

   type Memcpy3_DParms is record
      Src_Array : CUDA_Array_t;
      Src_Pos   : Pos;
      Src_Ptr   : Pitched_Ptr;
      Dst_Array : CUDA_Array_t;
      Dst_Pos   : Pos;
      Dst_Ptr   : Pitched_Ptr;
      Extent    : Extent_T;
      Kind      : Memcpy_Kind;
   end record;

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
   end record;

   type Memset_Params is record
      Dst          : System.Address;
      Pitch        : CUDA.Corecrt.Size_T;
      Value        : unsigned;
      Element_Size : unsigned;
      Width        : CUDA.Corecrt.Size_T;
      Height       : CUDA.Corecrt.Size_T;
   end record;

   type Host_Fn_T is access procedure (arg1 : System.Address);

   generic
      with procedure Temp_1 (Arg1 : System.Address);
   procedure Host_Fn_T_Gen (Arg1 : System.Address);

   type Host_Node_Params is record
      Fn        : Host_Fn_T;
      User_Data : System.Address;
   end record;

   type Stream_Capture_Status is
     (Stream_Capture_Status_None, Stream_Capture_Status_Active,
      Stream_Capture_Status_Invalidated);

   type Stream_Capture_Mode is
     (Stream_Capture_Mode_Global, Stream_Capture_Mode_Thread_Local,
      Stream_Capture_Mode_Relaxed);

   type Graphics_Resource is null record;
   subtype Graphics_Register_Flags is Interfaces.C.unsigned;
   Graphics_Register_Flags_None               : constant Interfaces.C.unsigned := 0;
   Graphics_Register_Flags_Read_Only          : constant Interfaces.C.unsigned := 1;
   Graphics_Register_Flags_Write_Discard      : constant Interfaces.C.unsigned := 2;
   Graphics_Register_Flags_Surface_Load_Store : constant Interfaces.C.unsigned := 4;
   Graphics_Register_Flags_Texture_Gather     : constant Interfaces.C.unsigned := 8;

   type Graphics_Map_Flags is
     (Graphics_Map_Flags_None, Graphics_Map_Flags_Read_Only,
      Graphics_Map_Flags_Write_Discard);

   type Graphics_Cube_Face is
     (Graphics_Cube_Face_Positive_X, Graphics_Cube_Face_Negative_X,
      Graphics_Cube_Face_Positive_Y, Graphics_Cube_Face_Negative_Y,
      Graphics_Cube_Face_Positive_Z, Graphics_Cube_Face_Negative_Z);

   type Resource_Type is
     (Resource_Type_Array, Resource_Type_Mipmapped_Array, Resource_Type_Linear,
      Resource_Type_Pitch2_D);

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
      Res_View_Format_Unsigned_Block_Compressed7);

   type Anon985_Struct987 is record
      C_Array : CUDA_Array_t;
   end record;

   type Anon985_Struct988 is record
      Mipmap : Mipmapped_Array_T;
   end record;

   type Anon985_Struct989 is record
      Dev_Ptr       : System.Address;
      Desc          : Channel_Format_Desc;
      Size_In_Bytes : CUDA.Corecrt.Size_T;
   end record;

   type Anon985_Struct990 is record
      Dev_Ptr        : System.Address;
      Desc           : Channel_Format_Desc;
      Width          : CUDA.Corecrt.Size_T;
      Height         : CUDA.Corecrt.Size_T;
      Pitch_In_Bytes : CUDA.Corecrt.Size_T;
   end record;

   type Anon985_Union986 is record
      C_Array  : Anon985_Struct987;
      Mipmap   : Anon985_Struct988;
      Linear   : Anon985_Struct989;
      Pitch2_D : Anon985_Struct990;
   end record;

   type Resource_Desc is record
      Res_Type : Resource_Type;
      Res      : Anon985_Union986;
   end record;

   type Resource_View_Desc is record
      Format             : Resource_View_Format;
      Width              : CUDA.Corecrt.Size_T;
      Height             : CUDA.Corecrt.Size_T;
      Depth              : CUDA.Corecrt.Size_T;
      First_Mipmap_Level : unsigned;
      Last_Mipmap_Level  : unsigned;
      First_Layer        : unsigned;
      Last_Layer         : unsigned;
   end record;

   type Pointer_Attributes is record
      Memory_Type    : Memory_Type_T;
      C_Type         : Memory_Type_T;
      Device         : int;
      Device_Pointer : System.Address;
      Host_Pointer   : System.Address;
      Is_Managed     : int;
   end record;

   type Func_Attributes is record
      Shared_Size_Bytes             : CUDA.Corecrt.Size_T;
      Const_Size_Bytes              : CUDA.Corecrt.Size_T;
      Local_Size_Bytes              : CUDA.Corecrt.Size_T;
      Max_Threads_Per_Block         : int;
      Num_Regs                      : int;
      Ptx_Version                   : int;
      Binary_Version                : int;
      Cache_Mode_CA                 : int;
      Max_Dynamic_Shared_Size_Bytes : int;
      Preferred_Shmem_Carveout      : int;
   end record;
   subtype Func_Attribute is unsigned;
   Func_Attribute_Max_Dynamic_Shared_Memory_Size   : unsigned;
   Func_Attribute_Preferred_Shared_Memory_Carveout : unsigned;
   Func_Attribute_Max                              : unsigned;

   type Func_Cache is
     (Func_Cache_Prefer_None, Func_Cache_Prefer_Shared, Func_Cache_Prefer_L1,
      Func_Cache_Prefer_Equal);

   type Shared_Mem_Config is
     (Shared_Mem_Bank_Size_Default, Shared_Mem_Bank_Size_Four_Byte,
      Shared_Mem_Bank_Size_Eight_Byte);
   subtype Shared_Carveout is int;
   Sharedmem_Carveout_Default    : int;
   Sharedmem_Carveout_Max_Shared : int;
   Sharedmem_Carveout_Max_L1     : int;

   type Compute_Mode is
     (Compute_Mode_Default, Compute_Mode_Exclusive, Compute_Mode_Prohibited,
      Compute_Mode_Exclusive_Process);

   type Limit is
     (Limit_Stack_Size, Limit_Printf_Fifo_Size, Limit_Malloc_Heap_Size,
      Limit_Dev_Runtime_Sync_Depth, Limit_Dev_Runtime_Pending_Launch_Count,
      Limit_Max_L2_Fetch_Granularity);
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

   type Output_Mode is (Key_Value_Pair, CSV);
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
   subtype Device_P2_PAttr is unsigned;
   Dev_P2_PAttr_Performance_Rank            : unsigned;
   Dev_P2_PAttr_Access_Supported            : unsigned;
   Dev_P2_PAttr_Native_Atomic_Supported     : unsigned;
   Dev_P2_PAttr_Cuda_Array_Access_Supported : unsigned;
   subtype Anon1005_Array1007 is Interfaces.C.char_array (0 .. 15);

   type CUuuid_St is record
      Bytes : Anon1005_Array1007;
   end record;
   subtype CUuuid is CUuuid_St;
   subtype UUID_T is CUuuid_St;
   subtype Anon1010_Array1012 is Interfaces.C.char_array (0 .. 255);
   subtype Anon1010_Array1014 is Interfaces.C.char_array (0 .. 7);

   type Anon1010_Array1016 is array (0 .. 2) of int;

   type Anon1010_Array1018 is array (0 .. 1) of int;

   type Device_Prop is record
      Name                                         : Anon1010_Array1012;
      Uuid                                         : UUID_T;
      Luid                                         : Anon1010_Array1014;
      Luid_Device_Node_Mask                        : unsigned;
      Total_Global_Mem                             : CUDA.Corecrt.Size_T;
      Shared_Mem_Per_Block                         : CUDA.Corecrt.Size_T;
      Regs_Per_Block                               : int;
      Warp_Size                                    : int;
      Mem_Pitch                                    : CUDA.Corecrt.Size_T;
      Max_Threads_Per_Block                        : int;
      Max_Threads_Dim                              : Anon1010_Array1016;
      Max_Grid_Size                                : Anon1010_Array1016;
      Clock_Rate                                   : int;
      Total_Const_Mem                              : CUDA.Corecrt.Size_T;
      Major                                        : int;
      Minor                                        : int;
      Texture_Alignment                            : CUDA.Corecrt.Size_T;
      Texture_Pitch_Alignment                      : CUDA.Corecrt.Size_T;
      Device_Overlap                               : int;
      Multi_Processor_Count                        : int;
      Kernel_Exec_Timeout_Enabled                  : int;
      Integrated                                   : int;
      Can_Map_Host_Memory                          : int;
      Compute_Mode                                 : int;
      Max_Texture1_D                               : int;
      Max_Texture1_DMipmap                         : int;
      Max_Texture1_DLinear                         : int;
      Max_Texture2_D                               : Anon1010_Array1018;
      Max_Texture2_DMipmap                         : Anon1010_Array1018;
      Max_Texture2_DLinear                         : Anon1010_Array1016;
      Max_Texture2_DGather                         : Anon1010_Array1018;
      Max_Texture3_D                               : Anon1010_Array1016;
      Max_Texture3_DAlt                            : Anon1010_Array1016;
      Max_Texture_Cubemap                          : int;
      Max_Texture1_DLayered                        : Anon1010_Array1018;
      Max_Texture2_DLayered                        : Anon1010_Array1016;
      Max_Texture_Cubemap_Layered                  : Anon1010_Array1018;
      Max_Surface1_D                               : int;
      Max_Surface2_D                               : Anon1010_Array1018;
      Max_Surface3_D                               : Anon1010_Array1016;
      Max_Surface1_DLayered                        : Anon1010_Array1018;
      Max_Surface2_DLayered                        : Anon1010_Array1016;
      Max_Surface_Cubemap                          : int;
      Max_Surface_Cubemap_Layered                  : Anon1010_Array1018;
      Surface_Alignment                            : CUDA.Corecrt.Size_T;
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
      Max_Threads_Per_Multi_Processor              : int;
      Stream_Priorities_Supported                  : int;
      Global_L1_Cache_Supported                    : int;
      Local_L1_Cache_Supported                     : int;
      Shared_Mem_Per_Multiprocessor                : CUDA.Corecrt.Size_T;
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
      Shared_Mem_Per_Block_Optin                   : CUDA.Corecrt.Size_T;
      Pageable_Memory_Access_Uses_Host_Page_Tables : int;
      Direct_Managed_Mem_Access_From_Host          : int;
   end record;
   subtype Anon1019_Array1021 is Interfaces.C.char_array (0 .. 63);

   type Ipc_Event_Handle_St is record
      Reserved : Anon1019_Array1021;
   end record;
   subtype Ipc_Event_Handle_T is Ipc_Event_Handle_St;
   subtype Anon1023_Array1021 is Interfaces.C.char_array (0 .. 63);

   type Ipc_Mem_Handle_St is record
      Reserved : Anon1023_Array1021;
   end record;
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

   type Anon1026_Struct1028 is record
      Handle : System.Address;
      Name   : System.Address;
   end record;

   type Anon1026_Union1027 is record
      Fd                : int;
      Win32             : Anon1026_Struct1028;
      Nv_Sci_Buf_Object : System.Address;
   end record;

   type External_Memory_Handle_Desc is record
      C_Type : External_Memory_Handle_Type;
      Handle : Anon1026_Union1027;
      Size   : Extensions.unsigned_long_long;
      Flags  : unsigned;
   end record;

   type External_Memory_Buffer_Desc is record
      Offset : Extensions.unsigned_long_long;
      Size   : Extensions.unsigned_long_long;
      Flags  : unsigned;
   end record;

   type External_Memory_Mipmapped_Array_Desc is record
      Offset      : Extensions.unsigned_long_long;
      Format_Desc : Channel_Format_Desc;
      Extent      : Extent_T;
      Flags       : unsigned;
      Num_Levels  : unsigned;
   end record;
   subtype External_Semaphore_Handle_Type is unsigned;
   External_Semaphore_Handle_Type_Opaque_Fd        : unsigned;
   External_Semaphore_Handle_Type_Opaque_Win32     : unsigned;
   External_Semaphore_Handle_Type_Opaque_Win32_Kmt : unsigned;
   External_Semaphore_Handle_Type_D3_D12_Fence     : unsigned;
   External_Semaphore_Handle_Type_D3_D11_Fence     : unsigned;
   External_Semaphore_Handle_Type_Nv_Sci_Sync      : unsigned;
   External_Semaphore_Handle_Type_Keyed_Mutex      : unsigned;
   External_Semaphore_Handle_Type_Keyed_Mutex_Kmt  : unsigned;

   type Anon1032_Struct1034 is record
      Handle : System.Address;
      Name   : System.Address;
   end record;

   type Anon1032_Union1033 is record
      Fd              : int;
      Win32           : Anon1032_Struct1034;
      Nv_Sci_Sync_Obj : System.Address;
   end record;

   type External_Semaphore_Handle_Desc is record
      C_Type : External_Semaphore_Handle_Type;
      Handle : Anon1032_Union1033;
      Flags  : unsigned;
   end record;

   type Anon1035_Struct1037 is record
      Value : Extensions.unsigned_long_long;
   end record;

   type Anon1035_Union1038 is record
      Fence    : System.Address;
      Reserved : Extensions.unsigned_long_long;
   end record;

   type Anon1035_Struct1039 is record
      Key : Extensions.unsigned_long_long;
   end record;

   type Anon1035_Struct1036 is record
      Fence       : Anon1035_Struct1037;
      Nv_Sci_Sync : Anon1035_Union1038;
      Keyed_Mutex : Anon1035_Struct1039;
   end record;

   type External_Semaphore_Signal_Params is record
      Params : Anon1035_Struct1036;
      Flags  : unsigned;
   end record;

   type Anon1040_Struct1042 is record
      Value : Extensions.unsigned_long_long;
   end record;

   type Anon1040_Union1043 is record
      Fence    : System.Address;
      Reserved : Extensions.unsigned_long_long;
   end record;

   type Anon1040_Struct1044 is record
      Key        : Extensions.unsigned_long_long;
      Timeout_Ms : unsigned;
   end record;

   type Anon1040_Struct1041 is record
      Fence       : Anon1040_Struct1042;
      Nv_Sci_Sync : Anon1040_Union1043;
      Keyed_Mutex : Anon1040_Struct1044;
   end record;

   type External_Semaphore_Wait_Params is record
      Params : Anon1040_Struct1041;
      Flags  : unsigned;
   end record;
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

   type CGScope is (CGScope_Invalid, CGScope_Grid, CGScope_Multi_Grid);

   type Launch_Params is record
      Func       : System.Address;
      Grid_Dim   : CUDA.Vector_Types.Dim3;
      Block_Dim  : CUDA.Vector_Types.Dim3;
      Args       : System.Address;
      Shared_Mem : CUDA.Corecrt.Size_T;
      Stream     : Stream_T;
   end record;

   type Kernel_Node_Params is record
      Func             : System.Address;
      Grid_Dim         : CUDA.Vector_Types.Dim3;
      Block_Dim        : CUDA.Vector_Types.Dim3;
      Shared_Mem_Bytes : unsigned;
      Kernel_Params    : System.Address;
      Extra            : System.Address;
   end record;

   type Graph_Node_Type is
     (Graph_Node_Type_Kernel, Graph_Node_Type_Memcpy, Graph_Node_Type_Memset,
      Graph_Node_Type_Host, Graph_Node_Type_Graph, Graph_Node_Type_Empty,
      Graph_Node_Type_Count);

   type CUgraph_Exec_St is null record;

   type Graph_Exec_T is access CUgraph_Exec_St;

   type Graph_Exec_Update_Result is
     (Graph_Exec_Update_Success, Graph_Exec_Update_Error,
      Graph_Exec_Update_Error_Topology_Changed,
      Graph_Exec_Update_Error_Node_Type_Changed,
      Graph_Exec_Update_Error_Function_Changed,
      Graph_Exec_Update_Error_Parameters_Changed,
      Graph_Exec_Update_Error_Not_Supported);
end CUDA.Driver_Types;
