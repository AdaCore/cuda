with CUDA.Crtdefs;
with CUDA.Vector_Types;
with Interfaces.C;
with Interfaces.C.Extensions;
with System;

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

   type CUDA_Array_T is access all CUDA_Array;

   type CUDA_Array_Const_T is access constant CUDA_Array;

   type Mipmapped_Array is null record;

   type Mipmapped_Array_T is access all Mipmapped_Array;

   type Mipmapped_Array_Const_T is access constant Mipmapped_Array;

   type Memory_Type_T is (Memory_Type_Unregistered, Memory_Type_Host, Memory_Type_Device, Memory_Type_Managed) with
      Convention => C;

   type Memcpy_Kind is (Memcpy_Host_To_Host, Memcpy_Host_To_Device, Memcpy_Device_To_Host, Memcpy_Device_To_Device, Memcpy_Default) with
      Convention => C;

   type Pitched_Ptr is record
      Ptr   : System.Address;
      Pitch : aliased CUDA.Crtdefs.Size_T;
      Xsize : aliased CUDA.Crtdefs.Size_T;
      Ysize : aliased CUDA.Crtdefs.Size_T;
   end record with
      Convention => C_Pass_By_Copy;

   type Extent_T is record
      Width  : aliased CUDA.Crtdefs.Size_T;
      Height : aliased CUDA.Crtdefs.Size_T;
      Depth  : aliased CUDA.Crtdefs.Size_T;
   end record with
      Convention => C_Pass_By_Copy;

   type Pos is record
      X : aliased CUDA.Crtdefs.Size_T;
      Y : aliased CUDA.Crtdefs.Size_T;
      Z : aliased CUDA.Crtdefs.Size_T;
   end record with
      Convention => C_Pass_By_Copy;

   type Memcpy3_DParms is record
      Src_Array : CUDA_Array_T;
      Src_Pos   : aliased Pos;
      Src_Ptr   : aliased Pitched_Ptr;
      Dst_Array : CUDA_Array_T;
      Dst_Pos   : aliased Pos;
      Dst_Ptr   : aliased Pitched_Ptr;
      Extent    : aliased Extent_T;
      Kind      : aliased Memcpy_Kind;
   end record with
      Convention => C_Pass_By_Copy;

   type Memcpy3_DPeer_Parms is record
      Src_Array  : CUDA_Array_T;
      Src_Pos    : aliased Pos;
      Src_Ptr    : aliased Pitched_Ptr;
      Src_Device : aliased Interfaces.C.int;
      Dst_Array  : CUDA_Array_T;
      Dst_Pos    : aliased Pos;
      Dst_Ptr    : aliased Pitched_Ptr;
      Dst_Device : aliased Interfaces.C.int;
      Extent     : aliased Extent_T;
   end record with
      Convention => C_Pass_By_Copy;

   type Memset_Params is record
      Dst          : System.Address;
      Pitch        : aliased CUDA.Crtdefs.Size_T;
      Value        : aliased Interfaces.C.unsigned;
      Element_Size : aliased Interfaces.C.unsigned;
      Width        : aliased CUDA.Crtdefs.Size_T;
      Height       : aliased CUDA.Crtdefs.Size_T;
   end record with
      Convention => C_Pass_By_Copy;

   type Host_Fn_T is access procedure (Arg1 : System.Address) with
      Convention => C;

   type Host_Node_Params is record
      Fn        : Host_Fn_T;
      User_Data : System.Address;
   end record with
      Convention => C_Pass_By_Copy;

   type Stream_Capture_Status is (Stream_Capture_Status_None, Stream_Capture_Status_Active, Stream_Capture_Status_Invalidated) with
      Convention => C;

   type Stream_Capture_Mode is (Stream_Capture_Mode_Global, Stream_Capture_Mode_Thread_Local, Stream_Capture_Mode_Relaxed) with
      Convention => C;

   type Graphics_Resource is null record;
   subtype Graphics_Register_Flags is Interfaces.C.unsigned;
   Graphics_Register_Flags_None               : constant Interfaces.C.unsigned := 0;
   Graphics_Register_Flags_Read_Only          : constant Interfaces.C.unsigned := 1;
   Graphics_Register_Flags_Write_Discard      : constant Interfaces.C.unsigned := 2;
   Graphics_Register_Flags_Surface_Load_Store : constant Interfaces.C.unsigned := 4;
   Graphics_Register_Flags_Texture_Gather     : constant Interfaces.C.unsigned := 8;

   type Graphics_Map_Flags is (Graphics_Map_Flags_None, Graphics_Map_Flags_Read_Only, Graphics_Map_Flags_Write_Discard) with
      Convention => C;

   type Graphics_Cube_Face is (Graphics_Cube_Face_Positive_X, Graphics_Cube_Face_Negative_X, Graphics_Cube_Face_Positive_Y, Graphics_Cube_Face_Negative_Y, Graphics_Cube_Face_Positive_Z, Graphics_Cube_Face_Negative_Z) with
      Convention => C;

   type Resource_Type is (Resource_Type_Array, Resource_Type_Mipmapped_Array, Resource_Type_Linear, Resource_Type_Pitch2_D) with
      Convention => C;

   type Resource_View_Format is
     (Res_View_Format_None, Res_View_Format_Unsigned_Char1, Res_View_Format_Unsigned_Char2, Res_View_Format_Unsigned_Char4, Res_View_Format_Signed_Char1, Res_View_Format_Signed_Char2, Res_View_Format_Signed_Char4, Res_View_Format_Unsigned_Short1, Res_View_Format_Unsigned_Short2,
      Res_View_Format_Unsigned_Short4, Res_View_Format_Signed_Short1, Res_View_Format_Signed_Short2, Res_View_Format_Signed_Short4, Res_View_Format_Unsigned_Int1, Res_View_Format_Unsigned_Int2, Res_View_Format_Unsigned_Int4, Res_View_Format_Signed_Int1, Res_View_Format_Signed_Int2,
      Res_View_Format_Signed_Int4, Res_View_Format_Half1, Res_View_Format_Half2, Res_View_Format_Half4, Res_View_Format_Float1, Res_View_Format_Float2, Res_View_Format_Float4, Res_View_Format_Unsigned_Block_Compressed1, Res_View_Format_Unsigned_Block_Compressed2,
      Res_View_Format_Unsigned_Block_Compressed3, Res_View_Format_Unsigned_Block_Compressed4, Res_View_Format_Signed_Block_Compressed4, Res_View_Format_Unsigned_Block_Compressed5, Res_View_Format_Signed_Block_Compressed5, Res_View_Format_Unsigned_Block_Compressed6_H,
      Res_View_Format_Signed_Block_Compressed6_H, Res_View_Format_Unsigned_Block_Compressed7) with
      Convention => C;

   type Anon985_C_Array_Struct is record
      C_Array : CUDA_Array_T;
   end record with
      Convention => C_Pass_By_Copy;

   type Anon985_Mipmap_Struct is record
      Mipmap : Mipmapped_Array_T;
   end record with
      Convention => C_Pass_By_Copy;

   type Anon985_Linear_Struct is record
      Dev_Ptr       : System.Address;
      Desc          : aliased Channel_Format_Desc;
      Size_In_Bytes : aliased CUDA.Crtdefs.Size_T;
   end record with
      Convention => C_Pass_By_Copy;

   type Anon985_Pitch2_D_Struct is record
      Dev_Ptr        : System.Address;
      Desc           : aliased Channel_Format_Desc;
      Width          : aliased CUDA.Crtdefs.Size_T;
      Height         : aliased CUDA.Crtdefs.Size_T;
      Pitch_In_Bytes : aliased CUDA.Crtdefs.Size_T;
   end record with
      Convention => C_Pass_By_Copy;

   type Anon985_Res_Union (discr : Interfaces.C.unsigned := 0) is record
      case discr is
         when 0 =>
            C_Array : aliased Anon985_C_Array_Struct;

         when 1 =>
            Mipmap : aliased Anon985_Mipmap_Struct;

         when 2 =>
            Linear : aliased Anon985_Linear_Struct;

         when others =>
            Pitch2_D : aliased Anon985_Pitch2_D_Struct;
      end case;
   end record with
      Convention      => C_Pass_By_Copy,
      Unchecked_Union => True;

   type Resource_Desc is record
      Res_Type : aliased Resource_Type;
      Res      : aliased Anon985_Res_Union;
   end record with
      Convention => C_Pass_By_Copy;

   type Resource_View_Desc is record
      Format             : aliased Resource_View_Format;
      Width              : aliased CUDA.Crtdefs.Size_T;
      Height             : aliased CUDA.Crtdefs.Size_T;
      Depth              : aliased CUDA.Crtdefs.Size_T;
      First_Mipmap_Level : aliased Interfaces.C.unsigned;
      Last_Mipmap_Level  : aliased Interfaces.C.unsigned;
      First_Layer        : aliased Interfaces.C.unsigned;
      Last_Layer         : aliased Interfaces.C.unsigned;
   end record with
      Convention => C_Pass_By_Copy;

   type Pointer_Attributes is record
      Memory_Type    : aliased Memory_Type_T;
      C_Type         : aliased Memory_Type_T;
      Device         : aliased Interfaces.C.int;
      Device_Pointer : System.Address;
      Host_Pointer   : System.Address;
      Is_Managed     : aliased Interfaces.C.int;
   end record with
      Convention => C_Pass_By_Copy;

   type Func_Attributes is record
      Shared_Size_Bytes             : aliased CUDA.Crtdefs.Size_T;
      Const_Size_Bytes              : aliased CUDA.Crtdefs.Size_T;
      Local_Size_Bytes              : aliased CUDA.Crtdefs.Size_T;
      Max_Threads_Per_Block         : aliased Interfaces.C.int;
      Num_Regs                      : aliased Interfaces.C.int;
      Ptx_Version                   : aliased Interfaces.C.int;
      Binary_Version                : aliased Interfaces.C.int;
      Cache_Mode_CA                 : aliased Interfaces.C.int;
      Max_Dynamic_Shared_Size_Bytes : aliased Interfaces.C.int;
      Preferred_Shmem_Carveout      : aliased Interfaces.C.int;
   end record with
      Convention => C_Pass_By_Copy;
   subtype Func_Attribute is Interfaces.C.unsigned;
   Func_Attribute_Max_Dynamic_Shared_Memory_Size   : constant Interfaces.C.unsigned := 8;
   Func_Attribute_Preferred_Shared_Memory_Carveout : constant Interfaces.C.unsigned := 9;
   Func_Attribute_Max                              : constant Interfaces.C.unsigned := 10;

   type Func_Cache is (Func_Cache_Prefer_None, Func_Cache_Prefer_Shared, Func_Cache_Prefer_L1, Func_Cache_Prefer_Equal) with
      Convention => C;

   type Shared_Mem_Config is (Shared_Mem_Bank_Size_Default, Shared_Mem_Bank_Size_Four_Byte, Shared_Mem_Bank_Size_Eight_Byte) with
      Convention => C;

   type Compute_Mode is (Compute_Mode_Default, Compute_Mode_Exclusive, Compute_Mode_Prohibited, Compute_Mode_Exclusive_Process) with
      Convention => C;

   type Limit is (Limit_Stack_Size, Limit_Printf_Fifo_Size, Limit_Malloc_Heap_Size, Limit_Dev_Runtime_Sync_Depth, Limit_Dev_Runtime_Pending_Launch_Count, Limit_Max_L2_Fetch_Granularity) with
      Convention => C;
   subtype Memory_Advise is Interfaces.C.unsigned;
   Mem_Advise_Set_Read_Mostly          : constant Interfaces.C.unsigned := 1;
   Mem_Advise_Unset_Read_Mostly        : constant Interfaces.C.unsigned := 2;
   Mem_Advise_Set_Preferred_Location   : constant Interfaces.C.unsigned := 3;
   Mem_Advise_Unset_Preferred_Location : constant Interfaces.C.unsigned := 4;
   Mem_Advise_Set_Accessed_By          : constant Interfaces.C.unsigned := 5;
   Mem_Advise_Unset_Accessed_By        : constant Interfaces.C.unsigned := 6;
   subtype Mem_Range_Attribute is Interfaces.C.unsigned;
   Mem_Range_Attribute_Read_Mostly            : constant Interfaces.C.unsigned := 1;
   Mem_Range_Attribute_Preferred_Location     : constant Interfaces.C.unsigned := 2;
   Mem_Range_Attribute_Accessed_By            : constant Interfaces.C.unsigned := 3;
   Mem_Range_Attribute_Last_Prefetch_Location : constant Interfaces.C.unsigned := 4;

   type Output_Mode is (Key_Value_Pair, CSV) with
      Convention => C;
   subtype Device_Attr is Interfaces.C.unsigned;
   Dev_Attr_Max_Threads_Per_Block                        : constant Interfaces.C.unsigned := 1;
   Dev_Attr_Max_Block_Dim_X                              : constant Interfaces.C.unsigned := 2;
   Dev_Attr_Max_Block_Dim_Y                              : constant Interfaces.C.unsigned := 3;
   Dev_Attr_Max_Block_Dim_Z                              : constant Interfaces.C.unsigned := 4;
   Dev_Attr_Max_Grid_Dim_X                               : constant Interfaces.C.unsigned := 5;
   Dev_Attr_Max_Grid_Dim_Y                               : constant Interfaces.C.unsigned := 6;
   Dev_Attr_Max_Grid_Dim_Z                               : constant Interfaces.C.unsigned := 7;
   Dev_Attr_Max_Shared_Memory_Per_Block                  : constant Interfaces.C.unsigned := 8;
   Dev_Attr_Total_Constant_Memory                        : constant Interfaces.C.unsigned := 9;
   Dev_Attr_Warp_Size                                    : constant Interfaces.C.unsigned := 10;
   Dev_Attr_Max_Pitch                                    : constant Interfaces.C.unsigned := 11;
   Dev_Attr_Max_Registers_Per_Block                      : constant Interfaces.C.unsigned := 12;
   Dev_Attr_Clock_Rate                                   : constant Interfaces.C.unsigned := 13;
   Dev_Attr_Texture_Alignment                            : constant Interfaces.C.unsigned := 14;
   Dev_Attr_Gpu_Overlap                                  : constant Interfaces.C.unsigned := 15;
   Dev_Attr_Multi_Processor_Count                        : constant Interfaces.C.unsigned := 16;
   Dev_Attr_Kernel_Exec_Timeout                          : constant Interfaces.C.unsigned := 17;
   Dev_Attr_Integrated                                   : constant Interfaces.C.unsigned := 18;
   Dev_Attr_Can_Map_Host_Memory                          : constant Interfaces.C.unsigned := 19;
   Dev_Attr_Compute_Mode                                 : constant Interfaces.C.unsigned := 20;
   Dev_Attr_Max_Texture1_DWidth                          : constant Interfaces.C.unsigned := 21;
   Dev_Attr_Max_Texture2_DWidth                          : constant Interfaces.C.unsigned := 22;
   Dev_Attr_Max_Texture2_DHeight                         : constant Interfaces.C.unsigned := 23;
   Dev_Attr_Max_Texture3_DWidth                          : constant Interfaces.C.unsigned := 24;
   Dev_Attr_Max_Texture3_DHeight                         : constant Interfaces.C.unsigned := 25;
   Dev_Attr_Max_Texture3_DDepth                          : constant Interfaces.C.unsigned := 26;
   Dev_Attr_Max_Texture2_DLayered_Width                  : constant Interfaces.C.unsigned := 27;
   Dev_Attr_Max_Texture2_DLayered_Height                 : constant Interfaces.C.unsigned := 28;
   Dev_Attr_Max_Texture2_DLayered_Layers                 : constant Interfaces.C.unsigned := 29;
   Dev_Attr_Surface_Alignment                            : constant Interfaces.C.unsigned := 30;
   Dev_Attr_Concurrent_Kernels                           : constant Interfaces.C.unsigned := 31;
   Dev_Attr_Ecc_Enabled                                  : constant Interfaces.C.unsigned := 32;
   Dev_Attr_Pci_Bus_Id                                   : constant Interfaces.C.unsigned := 33;
   Dev_Attr_Pci_Device_Id                                : constant Interfaces.C.unsigned := 34;
   Dev_Attr_Tcc_Driver                                   : constant Interfaces.C.unsigned := 35;
   Dev_Attr_Memory_Clock_Rate                            : constant Interfaces.C.unsigned := 36;
   Dev_Attr_Global_Memory_Bus_Width                      : constant Interfaces.C.unsigned := 37;
   Dev_Attr_L2_Cache_Size                                : constant Interfaces.C.unsigned := 38;
   Dev_Attr_Max_Threads_Per_Multi_Processor              : constant Interfaces.C.unsigned := 39;
   Dev_Attr_Async_Engine_Count                           : constant Interfaces.C.unsigned := 40;
   Dev_Attr_Unified_Addressing                           : constant Interfaces.C.unsigned := 41;
   Dev_Attr_Max_Texture1_DLayered_Width                  : constant Interfaces.C.unsigned := 42;
   Dev_Attr_Max_Texture1_DLayered_Layers                 : constant Interfaces.C.unsigned := 43;
   Dev_Attr_Max_Texture2_DGather_Width                   : constant Interfaces.C.unsigned := 45;
   Dev_Attr_Max_Texture2_DGather_Height                  : constant Interfaces.C.unsigned := 46;
   Dev_Attr_Max_Texture3_DWidth_Alt                      : constant Interfaces.C.unsigned := 47;
   Dev_Attr_Max_Texture3_DHeight_Alt                     : constant Interfaces.C.unsigned := 48;
   Dev_Attr_Max_Texture3_DDepth_Alt                      : constant Interfaces.C.unsigned := 49;
   Dev_Attr_Pci_Domain_Id                                : constant Interfaces.C.unsigned := 50;
   Dev_Attr_Texture_Pitch_Alignment                      : constant Interfaces.C.unsigned := 51;
   Dev_Attr_Max_Texture_Cubemap_Width                    : constant Interfaces.C.unsigned := 52;
   Dev_Attr_Max_Texture_Cubemap_Layered_Width            : constant Interfaces.C.unsigned := 53;
   Dev_Attr_Max_Texture_Cubemap_Layered_Layers           : constant Interfaces.C.unsigned := 54;
   Dev_Attr_Max_Surface1_DWidth                          : constant Interfaces.C.unsigned := 55;
   Dev_Attr_Max_Surface2_DWidth                          : constant Interfaces.C.unsigned := 56;
   Dev_Attr_Max_Surface2_DHeight                         : constant Interfaces.C.unsigned := 57;
   Dev_Attr_Max_Surface3_DWidth                          : constant Interfaces.C.unsigned := 58;
   Dev_Attr_Max_Surface3_DHeight                         : constant Interfaces.C.unsigned := 59;
   Dev_Attr_Max_Surface3_DDepth                          : constant Interfaces.C.unsigned := 60;
   Dev_Attr_Max_Surface1_DLayered_Width                  : constant Interfaces.C.unsigned := 61;
   Dev_Attr_Max_Surface1_DLayered_Layers                 : constant Interfaces.C.unsigned := 62;
   Dev_Attr_Max_Surface2_DLayered_Width                  : constant Interfaces.C.unsigned := 63;
   Dev_Attr_Max_Surface2_DLayered_Height                 : constant Interfaces.C.unsigned := 64;
   Dev_Attr_Max_Surface2_DLayered_Layers                 : constant Interfaces.C.unsigned := 65;
   Dev_Attr_Max_Surface_Cubemap_Width                    : constant Interfaces.C.unsigned := 66;
   Dev_Attr_Max_Surface_Cubemap_Layered_Width            : constant Interfaces.C.unsigned := 67;
   Dev_Attr_Max_Surface_Cubemap_Layered_Layers           : constant Interfaces.C.unsigned := 68;
   Dev_Attr_Max_Texture1_DLinear_Width                   : constant Interfaces.C.unsigned := 69;
   Dev_Attr_Max_Texture2_DLinear_Width                   : constant Interfaces.C.unsigned := 70;
   Dev_Attr_Max_Texture2_DLinear_Height                  : constant Interfaces.C.unsigned := 71;
   Dev_Attr_Max_Texture2_DLinear_Pitch                   : constant Interfaces.C.unsigned := 72;
   Dev_Attr_Max_Texture2_DMipmapped_Width                : constant Interfaces.C.unsigned := 73;
   Dev_Attr_Max_Texture2_DMipmapped_Height               : constant Interfaces.C.unsigned := 74;
   Dev_Attr_Compute_Capability_Major                     : constant Interfaces.C.unsigned := 75;
   Dev_Attr_Compute_Capability_Minor                     : constant Interfaces.C.unsigned := 76;
   Dev_Attr_Max_Texture1_DMipmapped_Width                : constant Interfaces.C.unsigned := 77;
   Dev_Attr_Stream_Priorities_Supported                  : constant Interfaces.C.unsigned := 78;
   Dev_Attr_Global_L1_Cache_Supported                    : constant Interfaces.C.unsigned := 79;
   Dev_Attr_Local_L1_Cache_Supported                     : constant Interfaces.C.unsigned := 80;
   Dev_Attr_Max_Shared_Memory_Per_Multiprocessor         : constant Interfaces.C.unsigned := 81;
   Dev_Attr_Max_Registers_Per_Multiprocessor             : constant Interfaces.C.unsigned := 82;
   Dev_Attr_Managed_Memory                               : constant Interfaces.C.unsigned := 83;
   Dev_Attr_Is_Multi_Gpu_Board                           : constant Interfaces.C.unsigned := 84;
   Dev_Attr_Multi_Gpu_Board_Group_ID                     : constant Interfaces.C.unsigned := 85;
   Dev_Attr_Host_Native_Atomic_Supported                 : constant Interfaces.C.unsigned := 86;
   Dev_Attr_Single_To_Double_Precision_Perf_Ratio        : constant Interfaces.C.unsigned := 87;
   Dev_Attr_Pageable_Memory_Access                       : constant Interfaces.C.unsigned := 88;
   Dev_Attr_Concurrent_Managed_Access                    : constant Interfaces.C.unsigned := 89;
   Dev_Attr_Compute_Preemption_Supported                 : constant Interfaces.C.unsigned := 90;
   Dev_Attr_Can_Use_Host_Pointer_For_Registered_Mem      : constant Interfaces.C.unsigned := 91;
   Dev_Attr_Reserved92                                   : constant Interfaces.C.unsigned := 92;
   Dev_Attr_Reserved93                                   : constant Interfaces.C.unsigned := 93;
   Dev_Attr_Reserved94                                   : constant Interfaces.C.unsigned := 94;
   Dev_Attr_Cooperative_Launch                           : constant Interfaces.C.unsigned := 95;
   Dev_Attr_Cooperative_Multi_Device_Launch              : constant Interfaces.C.unsigned := 96;
   Dev_Attr_Max_Shared_Memory_Per_Block_Optin            : constant Interfaces.C.unsigned := 97;
   Dev_Attr_Can_Flush_Remote_Writes                      : constant Interfaces.C.unsigned := 98;
   Dev_Attr_Host_Register_Supported                      : constant Interfaces.C.unsigned := 99;
   Dev_Attr_Pageable_Memory_Access_Uses_Host_Page_Tables : constant Interfaces.C.unsigned := 100;
   Dev_Attr_Direct_Managed_Mem_Access_From_Host          : constant Interfaces.C.unsigned := 101;
   subtype Device_P2_PAttr is Interfaces.C.unsigned;
   Dev_P2_PAttr_Performance_Rank            : constant Interfaces.C.unsigned := 1;
   Dev_P2_PAttr_Access_Supported            : constant Interfaces.C.unsigned := 2;
   Dev_P2_PAttr_Native_Atomic_Supported     : constant Interfaces.C.unsigned := 3;
   Dev_P2_PAttr_Cuda_Array_Access_Supported : constant Interfaces.C.unsigned := 4;
   subtype Anon1005_Bytes_Array is Interfaces.C.char_array (0 .. 15);

   type CUuuid_St is record
      Bytes : aliased Anon1005_Bytes_Array;
   end record with
      Convention => C_Pass_By_Copy;
   subtype CUuuid is CUuuid_St;
   subtype UUID_T is CUuuid_St;
   subtype Anon1010_Name_Array is Interfaces.C.char_array (0 .. 255);
   subtype Anon1010_Luid_Array is Interfaces.C.char_array (0 .. 7);

   type Anon1010_Max_Threads_Dim_Array is array (0 .. 2) of aliased Interfaces.C.int;

   type Anon1010_Max_Grid_Size_Array is array (0 .. 2) of aliased Interfaces.C.int;

   type Anon1010_Max_Texture2_D_Array is array (0 .. 1) of aliased Interfaces.C.int;

   type Anon1010_Max_Texture2_DMipmap_Array is array (0 .. 1) of aliased Interfaces.C.int;

   type Anon1010_Max_Texture2_DLinear_Array is array (0 .. 2) of aliased Interfaces.C.int;

   type Anon1010_Max_Texture2_DGather_Array is array (0 .. 1) of aliased Interfaces.C.int;

   type Anon1010_Max_Texture3_D_Array is array (0 .. 2) of aliased Interfaces.C.int;

   type Anon1010_Max_Texture3_DAlt_Array is array (0 .. 2) of aliased Interfaces.C.int;

   type Anon1010_Max_Texture1_DLayered_Array is array (0 .. 1) of aliased Interfaces.C.int;

   type Anon1010_Max_Texture2_DLayered_Array is array (0 .. 2) of aliased Interfaces.C.int;

   type Anon1010_Max_Texture_Cubemap_Layered_Array is array (0 .. 1) of aliased Interfaces.C.int;

   type Anon1010_Max_Surface2_D_Array is array (0 .. 1) of aliased Interfaces.C.int;

   type Anon1010_Max_Surface3_D_Array is array (0 .. 2) of aliased Interfaces.C.int;

   type Anon1010_Max_Surface1_DLayered_Array is array (0 .. 1) of aliased Interfaces.C.int;

   type Anon1010_Max_Surface2_DLayered_Array is array (0 .. 2) of aliased Interfaces.C.int;

   type Anon1010_Max_Surface_Cubemap_Layered_Array is array (0 .. 1) of aliased Interfaces.C.int;

   type Device_Prop is record
      Name                                         : aliased Anon1010_Name_Array;
      Uuid                                         : aliased UUID_T;
      Luid                                         : aliased Anon1010_Luid_Array;
      Luid_Device_Node_Mask                        : aliased Interfaces.C.unsigned;
      Total_Global_Mem                             : aliased CUDA.Crtdefs.Size_T;
      Shared_Mem_Per_Block                         : aliased CUDA.Crtdefs.Size_T;
      Regs_Per_Block                               : aliased Interfaces.C.int;
      Warp_Size                                    : aliased Interfaces.C.int;
      Mem_Pitch                                    : aliased CUDA.Crtdefs.Size_T;
      Max_Threads_Per_Block                        : aliased Interfaces.C.int;
      Max_Threads_Dim                              : aliased Anon1010_Max_Threads_Dim_Array;
      Max_Grid_Size                                : aliased Anon1010_Max_Grid_Size_Array;
      Clock_Rate                                   : aliased Interfaces.C.int;
      Total_Const_Mem                              : aliased CUDA.Crtdefs.Size_T;
      Major                                        : aliased Interfaces.C.int;
      Minor                                        : aliased Interfaces.C.int;
      Texture_Alignment                            : aliased CUDA.Crtdefs.Size_T;
      Texture_Pitch_Alignment                      : aliased CUDA.Crtdefs.Size_T;
      Device_Overlap                               : aliased Interfaces.C.int;
      Multi_Processor_Count                        : aliased Interfaces.C.int;
      Kernel_Exec_Timeout_Enabled                  : aliased Interfaces.C.int;
      Integrated                                   : aliased Interfaces.C.int;
      Can_Map_Host_Memory                          : aliased Interfaces.C.int;
      Compute_Mode                                 : aliased Interfaces.C.int;
      Max_Texture1_D                               : aliased Interfaces.C.int;
      Max_Texture1_DMipmap                         : aliased Interfaces.C.int;
      Max_Texture1_DLinear                         : aliased Interfaces.C.int;
      Max_Texture2_D                               : aliased Anon1010_Max_Texture2_D_Array;
      Max_Texture2_DMipmap                         : aliased Anon1010_Max_Texture2_DMipmap_Array;
      Max_Texture2_DLinear                         : aliased Anon1010_Max_Texture2_DLinear_Array;
      Max_Texture2_DGather                         : aliased Anon1010_Max_Texture2_DGather_Array;
      Max_Texture3_D                               : aliased Anon1010_Max_Texture3_D_Array;
      Max_Texture3_DAlt                            : aliased Anon1010_Max_Texture3_DAlt_Array;
      Max_Texture_Cubemap                          : aliased Interfaces.C.int;
      Max_Texture1_DLayered                        : aliased Anon1010_Max_Texture1_DLayered_Array;
      Max_Texture2_DLayered                        : aliased Anon1010_Max_Texture2_DLayered_Array;
      Max_Texture_Cubemap_Layered                  : aliased Anon1010_Max_Texture_Cubemap_Layered_Array;
      Max_Surface1_D                               : aliased Interfaces.C.int;
      Max_Surface2_D                               : aliased Anon1010_Max_Surface2_D_Array;
      Max_Surface3_D                               : aliased Anon1010_Max_Surface3_D_Array;
      Max_Surface1_DLayered                        : aliased Anon1010_Max_Surface1_DLayered_Array;
      Max_Surface2_DLayered                        : aliased Anon1010_Max_Surface2_DLayered_Array;
      Max_Surface_Cubemap                          : aliased Interfaces.C.int;
      Max_Surface_Cubemap_Layered                  : aliased Anon1010_Max_Surface_Cubemap_Layered_Array;
      Surface_Alignment                            : aliased CUDA.Crtdefs.Size_T;
      Concurrent_Kernels                           : aliased Interfaces.C.int;
      ECCEnabled                                   : aliased Interfaces.C.int;
      Pci_Bus_ID                                   : aliased Interfaces.C.int;
      Pci_Device_ID                                : aliased Interfaces.C.int;
      Pci_Domain_ID                                : aliased Interfaces.C.int;
      Tcc_Driver                                   : aliased Interfaces.C.int;
      Async_Engine_Count                           : aliased Interfaces.C.int;
      Unified_Addressing                           : aliased Interfaces.C.int;
      Memory_Clock_Rate                            : aliased Interfaces.C.int;
      Memory_Bus_Width                             : aliased Interfaces.C.int;
      L2_Cache_Size                                : aliased Interfaces.C.int;
      Max_Threads_Per_Multi_Processor              : aliased Interfaces.C.int;
      Stream_Priorities_Supported                  : aliased Interfaces.C.int;
      Global_L1_Cache_Supported                    : aliased Interfaces.C.int;
      Local_L1_Cache_Supported                     : aliased Interfaces.C.int;
      Shared_Mem_Per_Multiprocessor                : aliased CUDA.Crtdefs.Size_T;
      Regs_Per_Multiprocessor                      : aliased Interfaces.C.int;
      Managed_Memory                               : aliased Interfaces.C.int;
      Is_Multi_Gpu_Board                           : aliased Interfaces.C.int;
      Multi_Gpu_Board_Group_ID                     : aliased Interfaces.C.int;
      Host_Native_Atomic_Supported                 : aliased Interfaces.C.int;
      Single_To_Double_Precision_Perf_Ratio        : aliased Interfaces.C.int;
      Pageable_Memory_Access                       : aliased Interfaces.C.int;
      Concurrent_Managed_Access                    : aliased Interfaces.C.int;
      Compute_Preemption_Supported                 : aliased Interfaces.C.int;
      Can_Use_Host_Pointer_For_Registered_Mem      : aliased Interfaces.C.int;
      Cooperative_Launch                           : aliased Interfaces.C.int;
      Cooperative_Multi_Device_Launch              : aliased Interfaces.C.int;
      Shared_Mem_Per_Block_Optin                   : aliased CUDA.Crtdefs.Size_T;
      Pageable_Memory_Access_Uses_Host_Page_Tables : aliased Interfaces.C.int;
      Direct_Managed_Mem_Access_From_Host          : aliased Interfaces.C.int;
   end record with
      Convention => C_Pass_By_Copy;
   subtype Anon1019_Reserved_Array is Interfaces.C.char_array (0 .. 63);

   type Ipc_Event_Handle_St is record
      Reserved : aliased Anon1019_Reserved_Array;
   end record with
      Convention => C_Pass_By_Copy;
   subtype Ipc_Event_Handle_T is Ipc_Event_Handle_St;
   subtype Anon1023_Reserved_Array is Interfaces.C.char_array (0 .. 63);

   type Ipc_Mem_Handle_St is record
      Reserved : aliased Anon1023_Reserved_Array;
   end record with
      Convention => C_Pass_By_Copy;
   subtype Ipc_Mem_Handle_T is Ipc_Mem_Handle_St;
   subtype External_Memory_Handle_Type is Interfaces.C.unsigned;
   External_Memory_Handle_Type_Opaque_Fd           : constant Interfaces.C.unsigned := 1;
   External_Memory_Handle_Type_Opaque_Win32        : constant Interfaces.C.unsigned := 2;
   External_Memory_Handle_Type_Opaque_Win32_Kmt    : constant Interfaces.C.unsigned := 3;
   External_Memory_Handle_Type_D3_D12_Heap         : constant Interfaces.C.unsigned := 4;
   External_Memory_Handle_Type_D3_D12_Resource     : constant Interfaces.C.unsigned := 5;
   External_Memory_Handle_Type_D3_D11_Resource     : constant Interfaces.C.unsigned := 6;
   External_Memory_Handle_Type_D3_D11_Resource_Kmt : constant Interfaces.C.unsigned := 7;
   External_Memory_Handle_Type_Nv_Sci_Buf          : constant Interfaces.C.unsigned := 8;

   type Anon1026_Win32_Struct is record
      Handle : System.Address;
      Name   : System.Address;
   end record with
      Convention => C_Pass_By_Copy;

   type Anon1026_Handle_Union (discr : Interfaces.C.unsigned := 0) is record
      case discr is
         when 0 =>
            Fd : aliased Interfaces.C.int;

         when 1 =>
            Win32 : aliased Anon1026_Win32_Struct;

         when others =>
            Nv_Sci_Buf_Object : System.Address;
      end case;
   end record with
      Convention      => C_Pass_By_Copy,
      Unchecked_Union => True;

   type External_Memory_Handle_Desc is record
      C_Type : aliased External_Memory_Handle_Type;
      Handle : aliased Anon1026_Handle_Union;
      Size   : aliased Interfaces.C.Extensions.unsigned_long_long;
      Flags  : aliased Interfaces.C.unsigned;
   end record with
      Convention => C_Pass_By_Copy;

   type External_Memory_Buffer_Desc is record
      Offset : aliased Interfaces.C.Extensions.unsigned_long_long;
      Size   : aliased Interfaces.C.Extensions.unsigned_long_long;
      Flags  : aliased Interfaces.C.unsigned;
   end record with
      Convention => C_Pass_By_Copy;

   type External_Memory_Mipmapped_Array_Desc is record
      Offset      : aliased Interfaces.C.Extensions.unsigned_long_long;
      Format_Desc : aliased Channel_Format_Desc;
      Extent      : aliased Extent_T;
      Flags       : aliased Interfaces.C.unsigned;
      Num_Levels  : aliased Interfaces.C.unsigned;
   end record with
      Convention => C_Pass_By_Copy;
   subtype External_Semaphore_Handle_Type is Interfaces.C.unsigned;
   External_Semaphore_Handle_Type_Opaque_Fd        : constant Interfaces.C.unsigned := 1;
   External_Semaphore_Handle_Type_Opaque_Win32     : constant Interfaces.C.unsigned := 2;
   External_Semaphore_Handle_Type_Opaque_Win32_Kmt : constant Interfaces.C.unsigned := 3;
   External_Semaphore_Handle_Type_D3_D12_Fence     : constant Interfaces.C.unsigned := 4;
   External_Semaphore_Handle_Type_D3_D11_Fence     : constant Interfaces.C.unsigned := 5;
   External_Semaphore_Handle_Type_Nv_Sci_Sync      : constant Interfaces.C.unsigned := 6;
   External_Semaphore_Handle_Type_Keyed_Mutex      : constant Interfaces.C.unsigned := 7;
   External_Semaphore_Handle_Type_Keyed_Mutex_Kmt  : constant Interfaces.C.unsigned := 8;

   type Anon1032_Win32_Struct is record
      Handle : System.Address;
      Name   : System.Address;
   end record with
      Convention => C_Pass_By_Copy;

   type Anon1032_Handle_Union (discr : Interfaces.C.unsigned := 0) is record
      case discr is
         when 0 =>
            Fd : aliased Interfaces.C.int;

         when 1 =>
            Win32 : aliased Anon1032_Win32_Struct;

         when others =>
            Nv_Sci_Sync_Obj : System.Address;
      end case;
   end record with
      Convention      => C_Pass_By_Copy,
      Unchecked_Union => True;

   type External_Semaphore_Handle_Desc is record
      C_Type : aliased External_Semaphore_Handle_Type;
      Handle : aliased Anon1032_Handle_Union;
      Flags  : aliased Interfaces.C.unsigned;
   end record with
      Convention => C_Pass_By_Copy;

   type Anon1035_Fence_Struct is record
      Value : aliased Interfaces.C.Extensions.unsigned_long_long;
   end record with
      Convention => C_Pass_By_Copy;

   type Anon1035_Nv_Sci_Sync_Union (discr : Interfaces.C.unsigned := 0) is record
      case discr is
         when 0 =>
            Fence : System.Address;

         when others =>
            Reserved : aliased Interfaces.C.Extensions.unsigned_long_long;
      end case;
   end record with
      Convention      => C_Pass_By_Copy,
      Unchecked_Union => True;

   type Anon1035_Keyed_Mutex_Struct is record
      Key : aliased Interfaces.C.Extensions.unsigned_long_long;
   end record with
      Convention => C_Pass_By_Copy;

   type Anon1035_Params_Struct is record
      Fence       : aliased Anon1035_Fence_Struct;
      Nv_Sci_Sync : aliased Anon1035_Nv_Sci_Sync_Union;
      Keyed_Mutex : aliased Anon1035_Keyed_Mutex_Struct;
   end record with
      Convention => C_Pass_By_Copy;

   type External_Semaphore_Signal_Params is record
      Params : aliased Anon1035_Params_Struct;
      Flags  : aliased Interfaces.C.unsigned;
   end record with
      Convention => C_Pass_By_Copy;

   type Anon1040_Fence_Struct is record
      Value : aliased Interfaces.C.Extensions.unsigned_long_long;
   end record with
      Convention => C_Pass_By_Copy;

   type Anon1040_Nv_Sci_Sync_Union (discr : Interfaces.C.unsigned := 0) is record
      case discr is
         when 0 =>
            Fence : System.Address;

         when others =>
            Reserved : aliased Interfaces.C.Extensions.unsigned_long_long;
      end case;
   end record with
      Convention      => C_Pass_By_Copy,
      Unchecked_Union => True;

   type Anon1040_Keyed_Mutex_Struct is record
      Key        : aliased Interfaces.C.Extensions.unsigned_long_long;
      Timeout_Ms : aliased Interfaces.C.unsigned;
   end record with
      Convention => C_Pass_By_Copy;

   type Anon1040_Params_Struct is record
      Fence       : aliased Anon1040_Fence_Struct;
      Nv_Sci_Sync : aliased Anon1040_Nv_Sci_Sync_Union;
      Keyed_Mutex : aliased Anon1040_Keyed_Mutex_Struct;
   end record with
      Convention => C_Pass_By_Copy;

   type External_Semaphore_Wait_Params is record
      Params : aliased Anon1040_Params_Struct;
      Flags  : aliased Interfaces.C.unsigned;
   end record with
      Convention => C_Pass_By_Copy;
   subtype Error_T is Error;

   type CUstream_St is null record;

   type Stream_T is access all CUstream_St;

   type CUevent_St is null record;

   type Event_T is access all CUevent_St;

   type Graphics_Resource_T is access all Graphics_Resource;
   subtype Output_Mode_T is Output_Mode;

   type CUexternal_Memory_St is null record;

   type External_Memory_T is access all CUexternal_Memory_St;

   type CUexternal_Semaphore_St is null record;

   type External_Semaphore_T is access all CUexternal_Semaphore_St;

   type CUgraph_St is null record;

   type Graph_T is access all CUgraph_St;

   type CUgraph_Node_St is null record;

   type Graph_Node_T is access all CUgraph_Node_St;

   type CGScope is (CGScope_Invalid, CGScope_Grid, CGScope_Multi_Grid) with
      Convention => C;

   type Launch_Params is record
      Func       : System.Address;
      Grid_Dim   : aliased CUDA.Vector_Types.Dim3;
      Block_Dim  : aliased CUDA.Vector_Types.Dim3;
      Args       : System.Address;
      Shared_Mem : aliased CUDA.Crtdefs.Size_T;
      Stream     : Stream_T;
   end record with
      Convention => C_Pass_By_Copy;

   type Kernel_Node_Params is record
      Func             : System.Address;
      Grid_Dim         : aliased CUDA.Vector_Types.Dim3;
      Block_Dim        : aliased CUDA.Vector_Types.Dim3;
      Shared_Mem_Bytes : aliased Interfaces.C.unsigned;
      Kernel_Params    : System.Address;
      Extra            : System.Address;
   end record with
      Convention => C_Pass_By_Copy;

   type Graph_Node_Type is (Graph_Node_Type_Kernel, Graph_Node_Type_Memcpy, Graph_Node_Type_Memset, Graph_Node_Type_Host, Graph_Node_Type_Graph, Graph_Node_Type_Empty, Graph_Node_Type_Count) with
      Convention => C;

   type CUgraph_Exec_St is null record;

   type Graph_Exec_T is access all CUgraph_Exec_St;

   type Graph_Exec_Update_Result is
     (Graph_Exec_Update_Success, Graph_Exec_Update_Error, Graph_Exec_Update_Error_Topology_Changed, Graph_Exec_Update_Error_Node_Type_Changed, Graph_Exec_Update_Error_Function_Changed, Graph_Exec_Update_Error_Parameters_Changed, Graph_Exec_Update_Error_Not_Supported) with
      Convention => C;
end CUDA.Driver_Types;
