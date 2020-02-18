pragma Ada_2012;
pragma Style_Checks (Off);

with Interfaces.C; use Interfaces.C;
with System;
with crtdefs_h;
with Interfaces.C.Extensions;
with vector_types_h;

package driver_types_h is

   cudaHostAllocDefault : constant := 16#00#;  --  /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/driver_types.h:81
   cudaHostAllocPortable : constant := 16#01#;  --  /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/driver_types.h:82
   cudaHostAllocMapped : constant := 16#02#;  --  /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/driver_types.h:83
   cudaHostAllocWriteCombined : constant := 16#04#;  --  /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/driver_types.h:84

   cudaHostRegisterDefault : constant := 16#00#;  --  /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/driver_types.h:86
   cudaHostRegisterPortable : constant := 16#01#;  --  /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/driver_types.h:87
   cudaHostRegisterMapped : constant := 16#02#;  --  /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/driver_types.h:88
   cudaHostRegisterIoMemory : constant := 16#04#;  --  /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/driver_types.h:89

   cudaPeerAccessDefault : constant := 16#00#;  --  /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/driver_types.h:91

   cudaStreamDefault : constant := 16#00#;  --  /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/driver_types.h:93
   cudaStreamNonBlocking : constant := 16#01#;  --  /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/driver_types.h:94
   --  unsupported macro: cudaStreamLegacy ((cudaStream_t)0x1)
   --  unsupported macro: cudaStreamPerThread ((cudaStream_t)0x2)

   cudaEventDefault : constant := 16#00#;  --  /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/driver_types.h:116
   cudaEventBlockingSync : constant := 16#01#;  --  /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/driver_types.h:117
   cudaEventDisableTiming : constant := 16#02#;  --  /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/driver_types.h:118
   cudaEventInterprocess : constant := 16#04#;  --  /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/driver_types.h:119

   cudaDeviceScheduleAuto : constant := 16#00#;  --  /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/driver_types.h:121
   cudaDeviceScheduleSpin : constant := 16#01#;  --  /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/driver_types.h:122
   cudaDeviceScheduleYield : constant := 16#02#;  --  /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/driver_types.h:123
   cudaDeviceScheduleBlockingSync : constant := 16#04#;  --  /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/driver_types.h:124
   cudaDeviceBlockingSync : constant := 16#04#;  --  /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/driver_types.h:125

   cudaDeviceScheduleMask : constant := 16#07#;  --  /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/driver_types.h:128
   cudaDeviceMapHost : constant := 16#08#;  --  /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/driver_types.h:129
   cudaDeviceLmemResizeToMax : constant := 16#10#;  --  /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/driver_types.h:130
   cudaDeviceMask : constant := 16#1f#;  --  /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/driver_types.h:131

   cudaArrayDefault : constant := 16#00#;  --  /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/driver_types.h:133
   cudaArrayLayered : constant := 16#01#;  --  /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/driver_types.h:134
   cudaArraySurfaceLoadStore : constant := 16#02#;  --  /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/driver_types.h:135
   cudaArrayCubemap : constant := 16#04#;  --  /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/driver_types.h:136
   cudaArrayTextureGather : constant := 16#08#;  --  /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/driver_types.h:137
   cudaArrayColorAttachment : constant := 16#20#;  --  /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/driver_types.h:138

   cudaIpcMemLazyEnablePeerAccess : constant := 16#01#;  --  /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/driver_types.h:140

   cudaMemAttachGlobal : constant := 16#01#;  --  /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/driver_types.h:142
   cudaMemAttachHost : constant := 16#02#;  --  /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/driver_types.h:143
   cudaMemAttachSingle : constant := 16#04#;  --  /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/driver_types.h:144

   cudaOccupancyDefault : constant := 16#00#;  --  /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/driver_types.h:146
   cudaOccupancyDisableCachingOverride : constant := 16#01#;  --  /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/driver_types.h:147
   --  unsupported macro: cudaCpuDeviceId ((int)-1)
   --  unsupported macro: cudaInvalidDeviceId ((int)-2)

   cudaCooperativeLaunchMultiDeviceNoPreSync : constant := 16#01#;  --  /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/driver_types.h:157

   cudaCooperativeLaunchMultiDeviceNoPostSync : constant := 16#02#;  --  /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/driver_types.h:164
   --  unsupported macro: CUDART_CB __stdcall
   --  unsupported macro: cudaDevicePropDontCare { {'\0'}, {{0}}, {'\0'}, 0, 0, 0, 0, 0, 0, 0, {0, 0, 0}, {0, 0, 0}, 0, 0, -1, -1, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, {0, 0}, {0, 0}, {0, 0, 0}, {0, 0}, {0, 0, 0}, {0, 0, 0}, 0, {0, 0}, {0, 0, 0}, {0, 0}, 0, {0, 0}, {0, 0, 0}, {0, 0}, {0, 0, 0}, 0, {0, 0}, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, }

   CUDA_IPC_HANDLE_SIZE : constant := 64;  --  /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/driver_types.h:1821

   cudaExternalMemoryDedicated : constant := 16#1#;  --  /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/driver_types.h:1880

   cudaExternalSemaphoreSignalSkipNvSciBufMemSync : constant := 16#01#;  --  /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/driver_types.h:1889

   cudaExternalSemaphoreWaitSkipNvSciBufMemSync : constant := 16#02#;  --  /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/driver_types.h:1898

   cudaNvSciSyncAttrSignal : constant := 16#1#;  --  /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/driver_types.h:1905

   cudaNvSciSyncAttrWait : constant := 16#2#;  --  /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/driver_types.h:1912

   subtype cudaError is unsigned;
   cudaSuccess : constant unsigned := 0;
   cudaErrorInvalidValue : constant unsigned := 1;
   cudaErrorMemoryAllocation : constant unsigned := 2;
   cudaErrorInitializationError : constant unsigned := 3;
   cudaErrorCudartUnloading : constant unsigned := 4;
   cudaErrorProfilerDisabled : constant unsigned := 5;
   cudaErrorProfilerNotInitialized : constant unsigned := 6;
   cudaErrorProfilerAlreadyStarted : constant unsigned := 7;
   cudaErrorProfilerAlreadyStopped : constant unsigned := 8;
   cudaErrorInvalidConfiguration : constant unsigned := 9;
   cudaErrorInvalidPitchValue : constant unsigned := 12;
   cudaErrorInvalidSymbol : constant unsigned := 13;
   cudaErrorInvalidHostPointer : constant unsigned := 16;
   cudaErrorInvalidDevicePointer : constant unsigned := 17;
   cudaErrorInvalidTexture : constant unsigned := 18;
   cudaErrorInvalidTextureBinding : constant unsigned := 19;
   cudaErrorInvalidChannelDescriptor : constant unsigned := 20;
   cudaErrorInvalidMemcpyDirection : constant unsigned := 21;
   cudaErrorAddressOfConstant : constant unsigned := 22;
   cudaErrorTextureFetchFailed : constant unsigned := 23;
   cudaErrorTextureNotBound : constant unsigned := 24;
   cudaErrorSynchronizationError : constant unsigned := 25;
   cudaErrorInvalidFilterSetting : constant unsigned := 26;
   cudaErrorInvalidNormSetting : constant unsigned := 27;
   cudaErrorMixedDeviceExecution : constant unsigned := 28;
   cudaErrorNotYetImplemented : constant unsigned := 31;
   cudaErrorMemoryValueTooLarge : constant unsigned := 32;
   cudaErrorInsufficientDriver : constant unsigned := 35;
   cudaErrorInvalidSurface : constant unsigned := 37;
   cudaErrorDuplicateVariableName : constant unsigned := 43;
   cudaErrorDuplicateTextureName : constant unsigned := 44;
   cudaErrorDuplicateSurfaceName : constant unsigned := 45;
   cudaErrorDevicesUnavailable : constant unsigned := 46;
   cudaErrorIncompatibleDriverContext : constant unsigned := 49;
   cudaErrorMissingConfiguration : constant unsigned := 52;
   cudaErrorPriorLaunchFailure : constant unsigned := 53;
   cudaErrorLaunchMaxDepthExceeded : constant unsigned := 65;
   cudaErrorLaunchFileScopedTex : constant unsigned := 66;
   cudaErrorLaunchFileScopedSurf : constant unsigned := 67;
   cudaErrorSyncDepthExceeded : constant unsigned := 68;
   cudaErrorLaunchPendingCountExceeded : constant unsigned := 69;
   cudaErrorInvalidDeviceFunction : constant unsigned := 98;
   cudaErrorNoDevice : constant unsigned := 100;
   cudaErrorInvalidDevice : constant unsigned := 101;
   cudaErrorStartupFailure : constant unsigned := 127;
   cudaErrorInvalidKernelImage : constant unsigned := 200;
   cudaErrorDeviceUninitialized : constant unsigned := 201;
   cudaErrorMapBufferObjectFailed : constant unsigned := 205;
   cudaErrorUnmapBufferObjectFailed : constant unsigned := 206;
   cudaErrorArrayIsMapped : constant unsigned := 207;
   cudaErrorAlreadyMapped : constant unsigned := 208;
   cudaErrorNoKernelImageForDevice : constant unsigned := 209;
   cudaErrorAlreadyAcquired : constant unsigned := 210;
   cudaErrorNotMapped : constant unsigned := 211;
   cudaErrorNotMappedAsArray : constant unsigned := 212;
   cudaErrorNotMappedAsPointer : constant unsigned := 213;
   cudaErrorECCUncorrectable : constant unsigned := 214;
   cudaErrorUnsupportedLimit : constant unsigned := 215;
   cudaErrorDeviceAlreadyInUse : constant unsigned := 216;
   cudaErrorPeerAccessUnsupported : constant unsigned := 217;
   cudaErrorInvalidPtx : constant unsigned := 218;
   cudaErrorInvalidGraphicsContext : constant unsigned := 219;
   cudaErrorNvlinkUncorrectable : constant unsigned := 220;
   cudaErrorJitCompilerNotFound : constant unsigned := 221;
   cudaErrorInvalidSource : constant unsigned := 300;
   cudaErrorFileNotFound : constant unsigned := 301;
   cudaErrorSharedObjectSymbolNotFound : constant unsigned := 302;
   cudaErrorSharedObjectInitFailed : constant unsigned := 303;
   cudaErrorOperatingSystem : constant unsigned := 304;
   cudaErrorInvalidResourceHandle : constant unsigned := 400;
   cudaErrorIllegalState : constant unsigned := 401;
   cudaErrorSymbolNotFound : constant unsigned := 500;
   cudaErrorNotReady : constant unsigned := 600;
   cudaErrorIllegalAddress : constant unsigned := 700;
   cudaErrorLaunchOutOfResources : constant unsigned := 701;
   cudaErrorLaunchTimeout : constant unsigned := 702;
   cudaErrorLaunchIncompatibleTexturing : constant unsigned := 703;
   cudaErrorPeerAccessAlreadyEnabled : constant unsigned := 704;
   cudaErrorPeerAccessNotEnabled : constant unsigned := 705;
   cudaErrorSetOnActiveProcess : constant unsigned := 708;
   cudaErrorContextIsDestroyed : constant unsigned := 709;
   cudaErrorAssert : constant unsigned := 710;
   cudaErrorTooManyPeers : constant unsigned := 711;
   cudaErrorHostMemoryAlreadyRegistered : constant unsigned := 712;
   cudaErrorHostMemoryNotRegistered : constant unsigned := 713;
   cudaErrorHardwareStackError : constant unsigned := 714;
   cudaErrorIllegalInstruction : constant unsigned := 715;
   cudaErrorMisalignedAddress : constant unsigned := 716;
   cudaErrorInvalidAddressSpace : constant unsigned := 717;
   cudaErrorInvalidPc : constant unsigned := 718;
   cudaErrorLaunchFailure : constant unsigned := 719;
   cudaErrorCooperativeLaunchTooLarge : constant unsigned := 720;
   cudaErrorNotPermitted : constant unsigned := 800;
   cudaErrorNotSupported : constant unsigned := 801;
   cudaErrorSystemNotReady : constant unsigned := 802;
   cudaErrorSystemDriverMismatch : constant unsigned := 803;
   cudaErrorCompatNotSupportedOnDevice : constant unsigned := 804;
   cudaErrorStreamCaptureUnsupported : constant unsigned := 900;
   cudaErrorStreamCaptureInvalidated : constant unsigned := 901;
   cudaErrorStreamCaptureMerge : constant unsigned := 902;
   cudaErrorStreamCaptureUnmatched : constant unsigned := 903;
   cudaErrorStreamCaptureUnjoined : constant unsigned := 904;
   cudaErrorStreamCaptureIsolation : constant unsigned := 905;
   cudaErrorStreamCaptureImplicit : constant unsigned := 906;
   cudaErrorCapturedEvent : constant unsigned := 907;
   cudaErrorStreamCaptureWrongThread : constant unsigned := 908;
   cudaErrorTimeout : constant unsigned := 909;
   cudaErrorGraphExecUpdateFailure : constant unsigned := 910;
   cudaErrorUnknown : constant unsigned := 999;
   cudaErrorApiFailureBase : constant unsigned := 10000;  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/driver_types.h:189

   type cudaChannelFormatKind is 
     (cudaChannelFormatKindSigned,
      cudaChannelFormatKindUnsigned,
      cudaChannelFormatKindFloat,
      cudaChannelFormatKindNone)
   with Convention => C;  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/driver_types.h:973

   type cudaChannelFormatDesc is record
      x : aliased int;  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/driver_types.h:986
      y : aliased int;  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/driver_types.h:987
      z : aliased int;  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/driver_types.h:988
      w : aliased int;  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/driver_types.h:989
      f : aliased cudaChannelFormatKind;  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/driver_types.h:990
   end record
   with Convention => C_Pass_By_Copy;  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/driver_types.h:984

   type cudaArray is null record;   -- incomplete struct

   type cudaArray_t is access all cudaArray;  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/driver_types.h:996

   type cudaArray_const_t is access constant cudaArray;  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/driver_types.h:1001

   type cudaMipmappedArray is null record;   -- incomplete struct

   type cudaMipmappedArray_t is access all cudaMipmappedArray;  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/driver_types.h:1008

   type cudaMipmappedArray_const_t is access constant cudaMipmappedArray;  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/driver_types.h:1013

   type cudaMemoryType is 
     (cudaMemoryTypeUnregistered,
      cudaMemoryTypeHost,
      cudaMemoryTypeDevice,
      cudaMemoryTypeManaged)
   with Convention => C;  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/driver_types.h:1020

   type cudaMemcpyKind is 
     (cudaMemcpyHostToHost,
      cudaMemcpyHostToDevice,
      cudaMemcpyDeviceToHost,
      cudaMemcpyDeviceToDevice,
      cudaMemcpyDefault)
   with Convention => C;  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/driver_types.h:1031

   type cudaPitchedPtr is record
      ptr : System.Address;  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/driver_types.h:1047
      pitch : aliased crtdefs_h.size_t;  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/driver_types.h:1048
      xsize : aliased crtdefs_h.size_t;  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/driver_types.h:1049
      ysize : aliased crtdefs_h.size_t;  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/driver_types.h:1050
   end record
   with Convention => C_Pass_By_Copy;  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/driver_types.h:1045

   type cudaExtent is record
      width : aliased crtdefs_h.size_t;  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/driver_types.h:1060
      height : aliased crtdefs_h.size_t;  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/driver_types.h:1061
      depth : aliased crtdefs_h.size_t;  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/driver_types.h:1062
   end record
   with Convention => C_Pass_By_Copy;  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/driver_types.h:1058

   type cudaPos is record
      x : aliased crtdefs_h.size_t;  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/driver_types.h:1072
      y : aliased crtdefs_h.size_t;  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/driver_types.h:1073
      z : aliased crtdefs_h.size_t;  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/driver_types.h:1074
   end record
   with Convention => C_Pass_By_Copy;  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/driver_types.h:1070

   type cudaMemcpy3DParms is record
      srcArray : cudaArray_t;  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/driver_types.h:1082
      srcPos : aliased cudaPos;  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/driver_types.h:1083
      srcPtr : aliased cudaPitchedPtr;  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/driver_types.h:1084
      dstArray : cudaArray_t;  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/driver_types.h:1086
      dstPos : aliased cudaPos;  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/driver_types.h:1087
      dstPtr : aliased cudaPitchedPtr;  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/driver_types.h:1088
      extent : aliased cudaExtent;  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/driver_types.h:1090
      kind : aliased cudaMemcpyKind;  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/driver_types.h:1091
   end record
   with Convention => C_Pass_By_Copy;  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/driver_types.h:1080

   type cudaMemcpy3DPeerParms is record
      srcArray : cudaArray_t;  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/driver_types.h:1099
      srcPos : aliased cudaPos;  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/driver_types.h:1100
      srcPtr : aliased cudaPitchedPtr;  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/driver_types.h:1101
      srcDevice : aliased int;  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/driver_types.h:1102
      dstArray : cudaArray_t;  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/driver_types.h:1104
      dstPos : aliased cudaPos;  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/driver_types.h:1105
      dstPtr : aliased cudaPitchedPtr;  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/driver_types.h:1106
      dstDevice : aliased int;  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/driver_types.h:1107
      extent : aliased cudaExtent;  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/driver_types.h:1109
   end record
   with Convention => C_Pass_By_Copy;  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/driver_types.h:1097

   type cudaMemsetParams is record
      dst : System.Address;  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/driver_types.h:1116
      pitch : aliased crtdefs_h.size_t;  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/driver_types.h:1117
      value : aliased unsigned;  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/driver_types.h:1118
      elementSize : aliased unsigned;  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/driver_types.h:1119
      width : aliased crtdefs_h.size_t;  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/driver_types.h:1120
      height : aliased crtdefs_h.size_t;  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/driver_types.h:1121
   end record
   with Convention => C_Pass_By_Copy;  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/driver_types.h:1115

   type cudaHostFn_t is access procedure (arg1 : System.Address)
   with Convention => C;  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/driver_types.h:1134

   type cudaHostNodeParams is record
      fn : cudaHostFn_t;  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/driver_types.h:1140
      userData : System.Address;  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/driver_types.h:1141
   end record
   with Convention => C_Pass_By_Copy;  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/driver_types.h:1139

   type cudaStreamCaptureStatus is 
     (cudaStreamCaptureStatusNone,
      cudaStreamCaptureStatusActive,
      cudaStreamCaptureStatusInvalidated)
   with Convention => C;  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/driver_types.h:1147

   type cudaStreamCaptureMode is 
     (cudaStreamCaptureModeGlobal,
      cudaStreamCaptureModeThreadLocal,
      cudaStreamCaptureModeRelaxed)
   with Convention => C;  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/driver_types.h:1158

   type cudaGraphicsResource is null record;   -- incomplete struct

   subtype cudaGraphicsRegisterFlags is unsigned;
   cudaGraphicsRegisterFlagsNone : constant unsigned := 0;
   cudaGraphicsRegisterFlagsReadOnly : constant unsigned := 1;
   cudaGraphicsRegisterFlagsWriteDiscard : constant unsigned := 2;
   cudaGraphicsRegisterFlagsSurfaceLoadStore : constant unsigned := 4;
   cudaGraphicsRegisterFlagsTextureGather : constant unsigned := 8;  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/driver_types.h:1172

   type cudaGraphicsMapFlags is 
     (cudaGraphicsMapFlagsNone,
      cudaGraphicsMapFlagsReadOnly,
      cudaGraphicsMapFlagsWriteDiscard)
   with Convention => C;  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/driver_types.h:1184

   type cudaGraphicsCubeFace is 
     (cudaGraphicsCubeFacePositiveX,
      cudaGraphicsCubeFaceNegativeX,
      cudaGraphicsCubeFacePositiveY,
      cudaGraphicsCubeFaceNegativeY,
      cudaGraphicsCubeFacePositiveZ,
      cudaGraphicsCubeFaceNegativeZ)
   with Convention => C;  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/driver_types.h:1194

   type cudaResourceType is 
     (cudaResourceTypeArray,
      cudaResourceTypeMipmappedArray,
      cudaResourceTypeLinear,
      cudaResourceTypePitch2D)
   with Convention => C;  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/driver_types.h:1207

   type cudaResourceViewFormat is 
     (cudaResViewFormatNone,
      cudaResViewFormatUnsignedChar1,
      cudaResViewFormatUnsignedChar2,
      cudaResViewFormatUnsignedChar4,
      cudaResViewFormatSignedChar1,
      cudaResViewFormatSignedChar2,
      cudaResViewFormatSignedChar4,
      cudaResViewFormatUnsignedShort1,
      cudaResViewFormatUnsignedShort2,
      cudaResViewFormatUnsignedShort4,
      cudaResViewFormatSignedShort1,
      cudaResViewFormatSignedShort2,
      cudaResViewFormatSignedShort4,
      cudaResViewFormatUnsignedInt1,
      cudaResViewFormatUnsignedInt2,
      cudaResViewFormatUnsignedInt4,
      cudaResViewFormatSignedInt1,
      cudaResViewFormatSignedInt2,
      cudaResViewFormatSignedInt4,
      cudaResViewFormatHalf1,
      cudaResViewFormatHalf2,
      cudaResViewFormatHalf4,
      cudaResViewFormatFloat1,
      cudaResViewFormatFloat2,
      cudaResViewFormatFloat4,
      cudaResViewFormatUnsignedBlockCompressed1,
      cudaResViewFormatUnsignedBlockCompressed2,
      cudaResViewFormatUnsignedBlockCompressed3,
      cudaResViewFormatUnsignedBlockCompressed4,
      cudaResViewFormatSignedBlockCompressed4,
      cudaResViewFormatUnsignedBlockCompressed5,
      cudaResViewFormatSignedBlockCompressed5,
      cudaResViewFormatUnsignedBlockCompressed6H,
      cudaResViewFormatSignedBlockCompressed6H,
      cudaResViewFormatUnsignedBlockCompressed7)
   with Convention => C;  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/driver_types.h:1218

   type anon985_c_array_struct is record
      c_array : cudaArray_t;  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/driver_types.h:1265
   end record
   with Convention => C_Pass_By_Copy;
   type anon985_mipmap_struct is record
      mipmap : cudaMipmappedArray_t;  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/driver_types.h:1268
   end record
   with Convention => C_Pass_By_Copy;
   type anon985_linear_struct is record
      devPtr : System.Address;  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/driver_types.h:1271
      desc : aliased cudaChannelFormatDesc;  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/driver_types.h:1272
      sizeInBytes : aliased crtdefs_h.size_t;  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/driver_types.h:1273
   end record
   with Convention => C_Pass_By_Copy;
   type anon985_pitch2D_struct is record
      devPtr : System.Address;  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/driver_types.h:1276
      desc : aliased cudaChannelFormatDesc;  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/driver_types.h:1277
      width : aliased crtdefs_h.size_t;  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/driver_types.h:1278
      height : aliased crtdefs_h.size_t;  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/driver_types.h:1279
      pitchInBytes : aliased crtdefs_h.size_t;  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/driver_types.h:1280
   end record
   with Convention => C_Pass_By_Copy;
   type anon985_res_union (discr : unsigned := 0) is record
      case discr is
         when 0 =>
            c_array : aliased anon985_c_array_struct;  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/driver_types.h:1266
         when 1 =>
            mipmap : aliased anon985_mipmap_struct;  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/driver_types.h:1269
         when 2 =>
            linear : aliased anon985_linear_struct;  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/driver_types.h:1274
         when others =>
            pitch2D : aliased anon985_pitch2D_struct;  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/driver_types.h:1281
      end case;
   end record
   with Convention => C_Pass_By_Copy,
        Unchecked_Union => True;
   type cudaResourceDesc is record
      resType : aliased cudaResourceType;  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/driver_types.h:1261
      res : aliased anon985_res_union;  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/driver_types.h:1282
   end record
   with Convention => C_Pass_By_Copy;  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/driver_types.h:1260

   type cudaResourceViewDesc is record
      format : aliased cudaResourceViewFormat;  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/driver_types.h:1290
      width : aliased crtdefs_h.size_t;  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/driver_types.h:1291
      height : aliased crtdefs_h.size_t;  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/driver_types.h:1292
      depth : aliased crtdefs_h.size_t;  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/driver_types.h:1293
      firstMipmapLevel : aliased unsigned;  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/driver_types.h:1294
      lastMipmapLevel : aliased unsigned;  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/driver_types.h:1295
      firstLayer : aliased unsigned;  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/driver_types.h:1296
      lastLayer : aliased unsigned;  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/driver_types.h:1297
   end record
   with Convention => C_Pass_By_Copy;  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/driver_types.h:1288

   type cudaPointerAttributes is record
      memoryType : aliased cudaMemoryType;  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/driver_types.h:1313
      c_type : aliased cudaMemoryType;  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/driver_types.h:1319
      device : aliased int;  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/driver_types.h:1330
      devicePointer : System.Address;  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/driver_types.h:1336
      hostPointer : System.Address;  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/driver_types.h:1345
      isManaged : aliased int;  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/driver_types.h:1352
   end record
   with Convention => C_Pass_By_Copy;  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/driver_types.h:1303

   type cudaFuncAttributes is record
      sharedSizeBytes : aliased crtdefs_h.size_t;  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/driver_types.h:1365
      constSizeBytes : aliased crtdefs_h.size_t;  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/driver_types.h:1371
      localSizeBytes : aliased crtdefs_h.size_t;  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/driver_types.h:1376
      maxThreadsPerBlock : aliased int;  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/driver_types.h:1383
      numRegs : aliased int;  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/driver_types.h:1388
      ptxVersion : aliased int;  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/driver_types.h:1395
      binaryVersion : aliased int;  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/driver_types.h:1402
      cacheModeCA : aliased int;  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/driver_types.h:1408
      maxDynamicSharedSizeBytes : aliased int;  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/driver_types.h:1415
      preferredShmemCarveout : aliased int;  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/driver_types.h:1424
   end record
   with Convention => C_Pass_By_Copy;  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/driver_types.h:1358

   subtype cudaFuncAttribute is unsigned;
   cudaFuncAttributeMaxDynamicSharedMemorySize : constant unsigned := 8;
   cudaFuncAttributePreferredSharedMemoryCarveout : constant unsigned := 9;
   cudaFuncAttributeMax : constant unsigned := 10;  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/driver_types.h:1430

   type cudaFuncCache is 
     (cudaFuncCachePreferNone,
      cudaFuncCachePreferShared,
      cudaFuncCachePreferL1,
      cudaFuncCachePreferEqual)
   with Convention => C;  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/driver_types.h:1440

   type cudaSharedMemConfig is 
     (cudaSharedMemBankSizeDefault,
      cudaSharedMemBankSizeFourByte,
      cudaSharedMemBankSizeEightByte)
   with Convention => C;  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/driver_types.h:1452

   subtype cudaSharedCarveout is int;
   cudaSharedmemCarveoutDefault : constant int := -1;
   cudaSharedmemCarveoutMaxShared : constant int := 100;
   cudaSharedmemCarveoutMaxL1 : constant int := 0;  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/driver_types.h:1462

   type cudaComputeMode is 
     (cudaComputeModeDefault,
      cudaComputeModeExclusive,
      cudaComputeModeProhibited,
      cudaComputeModeExclusiveProcess)
   with Convention => C;  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/driver_types.h:1471

   type cudaLimit is 
     (cudaLimitStackSize,
      cudaLimitPrintfFifoSize,
      cudaLimitMallocHeapSize,
      cudaLimitDevRuntimeSyncDepth,
      cudaLimitDevRuntimePendingLaunchCount,
      cudaLimitMaxL2FetchGranularity)
   with Convention => C;  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/driver_types.h:1482

   subtype cudaMemoryAdvise is unsigned;
   cudaMemAdviseSetReadMostly : constant unsigned := 1;
   cudaMemAdviseUnsetReadMostly : constant unsigned := 2;
   cudaMemAdviseSetPreferredLocation : constant unsigned := 3;
   cudaMemAdviseUnsetPreferredLocation : constant unsigned := 4;
   cudaMemAdviseSetAccessedBy : constant unsigned := 5;
   cudaMemAdviseUnsetAccessedBy : constant unsigned := 6;  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/driver_types.h:1495

   subtype cudaMemRangeAttribute is unsigned;
   cudaMemRangeAttributeReadMostly : constant unsigned := 1;
   cudaMemRangeAttributePreferredLocation : constant unsigned := 2;
   cudaMemRangeAttributeAccessedBy : constant unsigned := 3;
   cudaMemRangeAttributeLastPrefetchLocation : constant unsigned := 4;  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/driver_types.h:1508

   type cudaOutputMode is 
     (cudaKeyValuePair,
      cudaCSV)
   with Convention => C;  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/driver_types.h:1519

   subtype cudaDeviceAttr is unsigned;
   cudaDevAttrMaxThreadsPerBlock : constant unsigned := 1;
   cudaDevAttrMaxBlockDimX : constant unsigned := 2;
   cudaDevAttrMaxBlockDimY : constant unsigned := 3;
   cudaDevAttrMaxBlockDimZ : constant unsigned := 4;
   cudaDevAttrMaxGridDimX : constant unsigned := 5;
   cudaDevAttrMaxGridDimY : constant unsigned := 6;
   cudaDevAttrMaxGridDimZ : constant unsigned := 7;
   cudaDevAttrMaxSharedMemoryPerBlock : constant unsigned := 8;
   cudaDevAttrTotalConstantMemory : constant unsigned := 9;
   cudaDevAttrWarpSize : constant unsigned := 10;
   cudaDevAttrMaxPitch : constant unsigned := 11;
   cudaDevAttrMaxRegistersPerBlock : constant unsigned := 12;
   cudaDevAttrClockRate : constant unsigned := 13;
   cudaDevAttrTextureAlignment : constant unsigned := 14;
   cudaDevAttrGpuOverlap : constant unsigned := 15;
   cudaDevAttrMultiProcessorCount : constant unsigned := 16;
   cudaDevAttrKernelExecTimeout : constant unsigned := 17;
   cudaDevAttrIntegrated : constant unsigned := 18;
   cudaDevAttrCanMapHostMemory : constant unsigned := 19;
   cudaDevAttrComputeMode : constant unsigned := 20;
   cudaDevAttrMaxTexture1DWidth : constant unsigned := 21;
   cudaDevAttrMaxTexture2DWidth : constant unsigned := 22;
   cudaDevAttrMaxTexture2DHeight : constant unsigned := 23;
   cudaDevAttrMaxTexture3DWidth : constant unsigned := 24;
   cudaDevAttrMaxTexture3DHeight : constant unsigned := 25;
   cudaDevAttrMaxTexture3DDepth : constant unsigned := 26;
   cudaDevAttrMaxTexture2DLayeredWidth : constant unsigned := 27;
   cudaDevAttrMaxTexture2DLayeredHeight : constant unsigned := 28;
   cudaDevAttrMaxTexture2DLayeredLayers : constant unsigned := 29;
   cudaDevAttrSurfaceAlignment : constant unsigned := 30;
   cudaDevAttrConcurrentKernels : constant unsigned := 31;
   cudaDevAttrEccEnabled : constant unsigned := 32;
   cudaDevAttrPciBusId : constant unsigned := 33;
   cudaDevAttrPciDeviceId : constant unsigned := 34;
   cudaDevAttrTccDriver : constant unsigned := 35;
   cudaDevAttrMemoryClockRate : constant unsigned := 36;
   cudaDevAttrGlobalMemoryBusWidth : constant unsigned := 37;
   cudaDevAttrL2CacheSize : constant unsigned := 38;
   cudaDevAttrMaxThreadsPerMultiProcessor : constant unsigned := 39;
   cudaDevAttrAsyncEngineCount : constant unsigned := 40;
   cudaDevAttrUnifiedAddressing : constant unsigned := 41;
   cudaDevAttrMaxTexture1DLayeredWidth : constant unsigned := 42;
   cudaDevAttrMaxTexture1DLayeredLayers : constant unsigned := 43;
   cudaDevAttrMaxTexture2DGatherWidth : constant unsigned := 45;
   cudaDevAttrMaxTexture2DGatherHeight : constant unsigned := 46;
   cudaDevAttrMaxTexture3DWidthAlt : constant unsigned := 47;
   cudaDevAttrMaxTexture3DHeightAlt : constant unsigned := 48;
   cudaDevAttrMaxTexture3DDepthAlt : constant unsigned := 49;
   cudaDevAttrPciDomainId : constant unsigned := 50;
   cudaDevAttrTexturePitchAlignment : constant unsigned := 51;
   cudaDevAttrMaxTextureCubemapWidth : constant unsigned := 52;
   cudaDevAttrMaxTextureCubemapLayeredWidth : constant unsigned := 53;
   cudaDevAttrMaxTextureCubemapLayeredLayers : constant unsigned := 54;
   cudaDevAttrMaxSurface1DWidth : constant unsigned := 55;
   cudaDevAttrMaxSurface2DWidth : constant unsigned := 56;
   cudaDevAttrMaxSurface2DHeight : constant unsigned := 57;
   cudaDevAttrMaxSurface3DWidth : constant unsigned := 58;
   cudaDevAttrMaxSurface3DHeight : constant unsigned := 59;
   cudaDevAttrMaxSurface3DDepth : constant unsigned := 60;
   cudaDevAttrMaxSurface1DLayeredWidth : constant unsigned := 61;
   cudaDevAttrMaxSurface1DLayeredLayers : constant unsigned := 62;
   cudaDevAttrMaxSurface2DLayeredWidth : constant unsigned := 63;
   cudaDevAttrMaxSurface2DLayeredHeight : constant unsigned := 64;
   cudaDevAttrMaxSurface2DLayeredLayers : constant unsigned := 65;
   cudaDevAttrMaxSurfaceCubemapWidth : constant unsigned := 66;
   cudaDevAttrMaxSurfaceCubemapLayeredWidth : constant unsigned := 67;
   cudaDevAttrMaxSurfaceCubemapLayeredLayers : constant unsigned := 68;
   cudaDevAttrMaxTexture1DLinearWidth : constant unsigned := 69;
   cudaDevAttrMaxTexture2DLinearWidth : constant unsigned := 70;
   cudaDevAttrMaxTexture2DLinearHeight : constant unsigned := 71;
   cudaDevAttrMaxTexture2DLinearPitch : constant unsigned := 72;
   cudaDevAttrMaxTexture2DMipmappedWidth : constant unsigned := 73;
   cudaDevAttrMaxTexture2DMipmappedHeight : constant unsigned := 74;
   cudaDevAttrComputeCapabilityMajor : constant unsigned := 75;
   cudaDevAttrComputeCapabilityMinor : constant unsigned := 76;
   cudaDevAttrMaxTexture1DMipmappedWidth : constant unsigned := 77;
   cudaDevAttrStreamPrioritiesSupported : constant unsigned := 78;
   cudaDevAttrGlobalL1CacheSupported : constant unsigned := 79;
   cudaDevAttrLocalL1CacheSupported : constant unsigned := 80;
   cudaDevAttrMaxSharedMemoryPerMultiprocessor : constant unsigned := 81;
   cudaDevAttrMaxRegistersPerMultiprocessor : constant unsigned := 82;
   cudaDevAttrManagedMemory : constant unsigned := 83;
   cudaDevAttrIsMultiGpuBoard : constant unsigned := 84;
   cudaDevAttrMultiGpuBoardGroupID : constant unsigned := 85;
   cudaDevAttrHostNativeAtomicSupported : constant unsigned := 86;
   cudaDevAttrSingleToDoublePrecisionPerfRatio : constant unsigned := 87;
   cudaDevAttrPageableMemoryAccess : constant unsigned := 88;
   cudaDevAttrConcurrentManagedAccess : constant unsigned := 89;
   cudaDevAttrComputePreemptionSupported : constant unsigned := 90;
   cudaDevAttrCanUseHostPointerForRegisteredMem : constant unsigned := 91;
   cudaDevAttrReserved92 : constant unsigned := 92;
   cudaDevAttrReserved93 : constant unsigned := 93;
   cudaDevAttrReserved94 : constant unsigned := 94;
   cudaDevAttrCooperativeLaunch : constant unsigned := 95;
   cudaDevAttrCooperativeMultiDeviceLaunch : constant unsigned := 96;
   cudaDevAttrMaxSharedMemoryPerBlockOptin : constant unsigned := 97;
   cudaDevAttrCanFlushRemoteWrites : constant unsigned := 98;
   cudaDevAttrHostRegisterSupported : constant unsigned := 99;
   cudaDevAttrPageableMemoryAccessUsesHostPageTables : constant unsigned := 100;
   cudaDevAttrDirectManagedMemAccessFromHost : constant unsigned := 101;  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/driver_types.h:1528

   subtype cudaDeviceP2PAttr is unsigned;
   cudaDevP2PAttrPerformanceRank : constant unsigned := 1;
   cudaDevP2PAttrAccessSupported : constant unsigned := 2;
   cudaDevP2PAttrNativeAtomicSupported : constant unsigned := 3;
   cudaDevP2PAttrCudaArrayAccessSupported : constant unsigned := 4;  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/driver_types.h:1636

   subtype anon1005_bytes_array is Interfaces.C.char_array (0 .. 15);
   type CUuuid_st is record
      bytes : aliased anon1005_bytes_array;  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/driver_types.h:1649
   end record
   with Convention => C_Pass_By_Copy;  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/driver_types.h:1648

   subtype CUuuid is CUuuid_st;  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/driver_types.h:1651

   subtype cudaUUID_t is CUuuid_st;  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/driver_types.h:1653

   subtype anon1010_name_array is Interfaces.C.char_array (0 .. 255);
   subtype anon1010_luid_array is Interfaces.C.char_array (0 .. 7);
   type anon1010_maxThreadsDim_array is array (0 .. 2) of aliased int;
   type anon1010_maxGridSize_array is array (0 .. 2) of aliased int;
   type anon1010_maxTexture2D_array is array (0 .. 1) of aliased int;
   type anon1010_maxTexture2DMipmap_array is array (0 .. 1) of aliased int;
   type anon1010_maxTexture2DLinear_array is array (0 .. 2) of aliased int;
   type anon1010_maxTexture2DGather_array is array (0 .. 1) of aliased int;
   type anon1010_maxTexture3D_array is array (0 .. 2) of aliased int;
   type anon1010_maxTexture3DAlt_array is array (0 .. 2) of aliased int;
   type anon1010_maxTexture1DLayered_array is array (0 .. 1) of aliased int;
   type anon1010_maxTexture2DLayered_array is array (0 .. 2) of aliased int;
   type anon1010_maxTextureCubemapLayered_array is array (0 .. 1) of aliased int;
   type anon1010_maxSurface2D_array is array (0 .. 1) of aliased int;
   type anon1010_maxSurface3D_array is array (0 .. 2) of aliased int;
   type anon1010_maxSurface1DLayered_array is array (0 .. 1) of aliased int;
   type anon1010_maxSurface2DLayered_array is array (0 .. 2) of aliased int;
   type anon1010_maxSurfaceCubemapLayered_array is array (0 .. 1) of aliased int;
   type cudaDeviceProp is record
      name : aliased anon1010_name_array;  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/driver_types.h:1660
      uuid : aliased cudaUUID_t;  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/driver_types.h:1661
      luid : aliased anon1010_luid_array;  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/driver_types.h:1662
      luidDeviceNodeMask : aliased unsigned;  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/driver_types.h:1663
      totalGlobalMem : aliased crtdefs_h.size_t;  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/driver_types.h:1664
      sharedMemPerBlock : aliased crtdefs_h.size_t;  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/driver_types.h:1665
      regsPerBlock : aliased int;  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/driver_types.h:1666
      warpSize : aliased int;  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/driver_types.h:1667
      memPitch : aliased crtdefs_h.size_t;  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/driver_types.h:1668
      maxThreadsPerBlock : aliased int;  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/driver_types.h:1669
      maxThreadsDim : aliased anon1010_maxThreadsDim_array;  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/driver_types.h:1670
      maxGridSize : aliased anon1010_maxGridSize_array;  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/driver_types.h:1671
      clockRate : aliased int;  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/driver_types.h:1672
      totalConstMem : aliased crtdefs_h.size_t;  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/driver_types.h:1673
      major : aliased int;  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/driver_types.h:1674
      minor : aliased int;  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/driver_types.h:1675
      textureAlignment : aliased crtdefs_h.size_t;  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/driver_types.h:1676
      texturePitchAlignment : aliased crtdefs_h.size_t;  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/driver_types.h:1677
      deviceOverlap : aliased int;  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/driver_types.h:1678
      multiProcessorCount : aliased int;  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/driver_types.h:1679
      kernelExecTimeoutEnabled : aliased int;  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/driver_types.h:1680
      integrated : aliased int;  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/driver_types.h:1681
      canMapHostMemory : aliased int;  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/driver_types.h:1682
      computeMode : aliased int;  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/driver_types.h:1683
      maxTexture1D : aliased int;  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/driver_types.h:1684
      maxTexture1DMipmap : aliased int;  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/driver_types.h:1685
      maxTexture1DLinear : aliased int;  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/driver_types.h:1686
      maxTexture2D : aliased anon1010_maxTexture2D_array;  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/driver_types.h:1687
      maxTexture2DMipmap : aliased anon1010_maxTexture2DMipmap_array;  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/driver_types.h:1688
      maxTexture2DLinear : aliased anon1010_maxTexture2DLinear_array;  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/driver_types.h:1689
      maxTexture2DGather : aliased anon1010_maxTexture2DGather_array;  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/driver_types.h:1690
      maxTexture3D : aliased anon1010_maxTexture3D_array;  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/driver_types.h:1691
      maxTexture3DAlt : aliased anon1010_maxTexture3DAlt_array;  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/driver_types.h:1692
      maxTextureCubemap : aliased int;  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/driver_types.h:1693
      maxTexture1DLayered : aliased anon1010_maxTexture1DLayered_array;  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/driver_types.h:1694
      maxTexture2DLayered : aliased anon1010_maxTexture2DLayered_array;  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/driver_types.h:1695
      maxTextureCubemapLayered : aliased anon1010_maxTextureCubemapLayered_array;  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/driver_types.h:1696
      maxSurface1D : aliased int;  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/driver_types.h:1697
      maxSurface2D : aliased anon1010_maxSurface2D_array;  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/driver_types.h:1698
      maxSurface3D : aliased anon1010_maxSurface3D_array;  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/driver_types.h:1699
      maxSurface1DLayered : aliased anon1010_maxSurface1DLayered_array;  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/driver_types.h:1700
      maxSurface2DLayered : aliased anon1010_maxSurface2DLayered_array;  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/driver_types.h:1701
      maxSurfaceCubemap : aliased int;  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/driver_types.h:1702
      maxSurfaceCubemapLayered : aliased anon1010_maxSurfaceCubemapLayered_array;  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/driver_types.h:1703
      surfaceAlignment : aliased crtdefs_h.size_t;  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/driver_types.h:1704
      concurrentKernels : aliased int;  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/driver_types.h:1705
      ECCEnabled : aliased int;  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/driver_types.h:1706
      pciBusID : aliased int;  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/driver_types.h:1707
      pciDeviceID : aliased int;  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/driver_types.h:1708
      pciDomainID : aliased int;  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/driver_types.h:1709
      tccDriver : aliased int;  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/driver_types.h:1710
      asyncEngineCount : aliased int;  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/driver_types.h:1711
      unifiedAddressing : aliased int;  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/driver_types.h:1712
      memoryClockRate : aliased int;  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/driver_types.h:1713
      memoryBusWidth : aliased int;  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/driver_types.h:1714
      l2CacheSize : aliased int;  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/driver_types.h:1715
      maxThreadsPerMultiProcessor : aliased int;  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/driver_types.h:1716
      streamPrioritiesSupported : aliased int;  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/driver_types.h:1717
      globalL1CacheSupported : aliased int;  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/driver_types.h:1718
      localL1CacheSupported : aliased int;  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/driver_types.h:1719
      sharedMemPerMultiprocessor : aliased crtdefs_h.size_t;  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/driver_types.h:1720
      regsPerMultiprocessor : aliased int;  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/driver_types.h:1721
      managedMemory : aliased int;  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/driver_types.h:1722
      isMultiGpuBoard : aliased int;  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/driver_types.h:1723
      multiGpuBoardGroupID : aliased int;  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/driver_types.h:1724
      hostNativeAtomicSupported : aliased int;  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/driver_types.h:1725
      singleToDoublePrecisionPerfRatio : aliased int;  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/driver_types.h:1726
      pageableMemoryAccess : aliased int;  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/driver_types.h:1727
      concurrentManagedAccess : aliased int;  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/driver_types.h:1728
      computePreemptionSupported : aliased int;  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/driver_types.h:1729
      canUseHostPointerForRegisteredMem : aliased int;  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/driver_types.h:1730
      cooperativeLaunch : aliased int;  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/driver_types.h:1731
      cooperativeMultiDeviceLaunch : aliased int;  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/driver_types.h:1732
      sharedMemPerBlockOptin : aliased crtdefs_h.size_t;  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/driver_types.h:1733
      pageableMemoryAccessUsesHostPageTables : aliased int;  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/driver_types.h:1734
      directManagedMemAccessFromHost : aliased int;  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/driver_types.h:1735
   end record
   with Convention => C_Pass_By_Copy;  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/driver_types.h:1658

   subtype anon1019_reserved_array is Interfaces.C.char_array (0 .. 63);
   type cudaIpcEventHandle_st is record
      reserved : aliased anon1019_reserved_array;  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/driver_types.h:1828
   end record
   with Convention => C_Pass_By_Copy;  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/driver_types.h:1826

   subtype cudaIpcEventHandle_t is cudaIpcEventHandle_st;  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/driver_types.h:1829

   subtype anon1023_reserved_array is Interfaces.C.char_array (0 .. 63);
   type cudaIpcMemHandle_st is record
      reserved : aliased anon1023_reserved_array;  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/driver_types.h:1836
   end record
   with Convention => C_Pass_By_Copy;  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/driver_types.h:1834

   subtype cudaIpcMemHandle_t is cudaIpcMemHandle_st;  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/driver_types.h:1837

   subtype cudaExternalMemoryHandleType is unsigned;
   cudaExternalMemoryHandleTypeOpaqueFd : constant unsigned := 1;
   cudaExternalMemoryHandleTypeOpaqueWin32 : constant unsigned := 2;
   cudaExternalMemoryHandleTypeOpaqueWin32Kmt : constant unsigned := 3;
   cudaExternalMemoryHandleTypeD3D12Heap : constant unsigned := 4;
   cudaExternalMemoryHandleTypeD3D12Resource : constant unsigned := 5;
   cudaExternalMemoryHandleTypeD3D11Resource : constant unsigned := 6;
   cudaExternalMemoryHandleTypeD3D11ResourceKmt : constant unsigned := 7;
   cudaExternalMemoryHandleTypeNvSciBuf : constant unsigned := 8;  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/driver_types.h:1842

   type anon1026_win32_struct is record
      handle : System.Address;  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/driver_types.h:1948
      name : System.Address;  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/driver_types.h:1953
   end record
   with Convention => C_Pass_By_Copy;
   type anon1026_handle_union (discr : unsigned := 0) is record
      case discr is
         when 0 =>
            fd : aliased int;  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/driver_types.h:1928
         when 1 =>
            win32 : aliased anon1026_win32_struct;  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/driver_types.h:1954
         when others =>
            nvSciBufObject : System.Address;  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/driver_types.h:1959
      end case;
   end record
   with Convention => C_Pass_By_Copy,
        Unchecked_Union => True;
   type cudaExternalMemoryHandleDesc is record
      c_type : aliased cudaExternalMemoryHandleType;  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/driver_types.h:1921
      handle : aliased anon1026_handle_union;  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/driver_types.h:1960
      size : aliased Extensions.unsigned_long_long;  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/driver_types.h:1964
      flags : aliased unsigned;  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/driver_types.h:1968
   end record
   with Convention => C_Pass_By_Copy;  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/driver_types.h:1917

   type cudaExternalMemoryBufferDesc is record
      offset : aliased Extensions.unsigned_long_long;  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/driver_types.h:1978
      size : aliased Extensions.unsigned_long_long;  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/driver_types.h:1982
      flags : aliased unsigned;  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/driver_types.h:1986
   end record
   with Convention => C_Pass_By_Copy;  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/driver_types.h:1974

   type cudaExternalMemoryMipmappedArrayDesc is record
      offset : aliased Extensions.unsigned_long_long;  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/driver_types.h:1997
      formatDesc : aliased cudaChannelFormatDesc;  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/driver_types.h:2001
      extent : aliased cudaExtent;  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/driver_types.h:2005
      flags : aliased unsigned;  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/driver_types.h:2010
      numLevels : aliased unsigned;  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/driver_types.h:2014
   end record
   with Convention => C_Pass_By_Copy;  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/driver_types.h:1992

   subtype cudaExternalSemaphoreHandleType is unsigned;
   cudaExternalSemaphoreHandleTypeOpaqueFd : constant unsigned := 1;
   cudaExternalSemaphoreHandleTypeOpaqueWin32 : constant unsigned := 2;
   cudaExternalSemaphoreHandleTypeOpaqueWin32Kmt : constant unsigned := 3;
   cudaExternalSemaphoreHandleTypeD3D12Fence : constant unsigned := 4;
   cudaExternalSemaphoreHandleTypeD3D11Fence : constant unsigned := 5;
   cudaExternalSemaphoreHandleTypeNvSciSync : constant unsigned := 6;
   cudaExternalSemaphoreHandleTypeKeyedMutex : constant unsigned := 7;
   cudaExternalSemaphoreHandleTypeKeyedMutexKmt : constant unsigned := 8;  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/driver_types.h:2020

   type anon1032_win32_struct is record
      handle : System.Address;  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/driver_types.h:2087
      name : System.Address;  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/driver_types.h:2092
   end record
   with Convention => C_Pass_By_Copy;
   type anon1032_handle_union (discr : unsigned := 0) is record
      case discr is
         when 0 =>
            fd : aliased int;  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/driver_types.h:2068
         when 1 =>
            win32 : aliased anon1032_win32_struct;  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/driver_types.h:2093
         when others =>
            nvSciSyncObj : System.Address;  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/driver_types.h:2097
      end case;
   end record
   with Convention => C_Pass_By_Copy,
        Unchecked_Union => True;
   type cudaExternalSemaphoreHandleDesc is record
      c_type : aliased cudaExternalSemaphoreHandleType;  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/driver_types.h:2062
      handle : aliased anon1032_handle_union;  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/driver_types.h:2098
      flags : aliased unsigned;  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/driver_types.h:2102
   end record
   with Convention => C_Pass_By_Copy;  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/driver_types.h:2058

   type anon1035_fence_struct is record
      value : aliased Extensions.unsigned_long_long;  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/driver_types.h:2117
   end record
   with Convention => C_Pass_By_Copy;
   type anon1035_nvSciSync_union (discr : unsigned := 0) is record
      case discr is
         when 0 =>
            fence : System.Address;  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/driver_types.h:2124
         when others =>
            reserved : aliased Extensions.unsigned_long_long;  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/driver_types.h:2125
      end case;
   end record
   with Convention => C_Pass_By_Copy,
        Unchecked_Union => True;
   type anon1035_keyedMutex_struct is record
      key : aliased Extensions.unsigned_long_long;  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/driver_types.h:2134
   end record
   with Convention => C_Pass_By_Copy;
   type anon1035_params_struct is record
      fence : aliased anon1035_fence_struct;  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/driver_types.h:2118
      nvSciSync : aliased anon1035_nvSciSync_union;  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/driver_types.h:2126
      keyedMutex : aliased anon1035_keyedMutex_struct;  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/driver_types.h:2135
   end record
   with Convention => C_Pass_By_Copy;
   type cudaExternalSemaphoreSignalParams is record
      params : aliased anon1035_params_struct;  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/driver_types.h:2136
      flags : aliased unsigned;  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/driver_types.h:2147
   end record
   with Convention => C_Pass_By_Copy;  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/driver_types.h:2108

   type anon1040_fence_struct is record
      value : aliased Extensions.unsigned_long_long;  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/driver_types.h:2162
   end record
   with Convention => C_Pass_By_Copy;
   type anon1040_nvSciSync_union (discr : unsigned := 0) is record
      case discr is
         when 0 =>
            fence : System.Address;  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/driver_types.h:2169
         when others =>
            reserved : aliased Extensions.unsigned_long_long;  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/driver_types.h:2170
      end case;
   end record
   with Convention => C_Pass_By_Copy,
        Unchecked_Union => True;
   type anon1040_keyedMutex_struct is record
      key : aliased Extensions.unsigned_long_long;  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/driver_types.h:2179
      timeoutMs : aliased unsigned;  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/driver_types.h:2183
   end record
   with Convention => C_Pass_By_Copy;
   type anon1040_params_struct is record
      fence : aliased anon1040_fence_struct;  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/driver_types.h:2163
      nvSciSync : aliased anon1040_nvSciSync_union;  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/driver_types.h:2171
      keyedMutex : aliased anon1040_keyedMutex_struct;  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/driver_types.h:2184
   end record
   with Convention => C_Pass_By_Copy;
   type cudaExternalSemaphoreWaitParams is record
      params : aliased anon1040_params_struct;  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/driver_types.h:2185
      flags : aliased unsigned;  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/driver_types.h:2196
   end record
   with Convention => C_Pass_By_Copy;  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/driver_types.h:2153

   subtype cudaError_t is cudaError;  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/driver_types.h:2209

   type CUstream_st is null record;   -- incomplete struct

   type cudaStream_t is access all CUstream_st;  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/driver_types.h:2214

   type CUevent_st is null record;   -- incomplete struct

   type cudaEvent_t is access all CUevent_st;  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/driver_types.h:2219

   type cudaGraphicsResource_t is access all cudaGraphicsResource;  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/driver_types.h:2224

   subtype cudaOutputMode_t is cudaOutputMode;  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/driver_types.h:2229

   type CUexternalMemory_st is null record;   -- incomplete struct

   type cudaExternalMemory_t is access all CUexternalMemory_st;  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/driver_types.h:2234

   type CUexternalSemaphore_st is null record;   -- incomplete struct

   type cudaExternalSemaphore_t is access all CUexternalSemaphore_st;  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/driver_types.h:2239

   type CUgraph_st is null record;   -- incomplete struct

   type cudaGraph_t is access all CUgraph_st;  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/driver_types.h:2244

   type CUgraphNode_st is null record;   -- incomplete struct

   type cudaGraphNode_t is access all CUgraphNode_st;  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/driver_types.h:2249

   type cudaCGScope is 
     (cudaCGScopeInvalid,
      cudaCGScopeGrid,
      cudaCGScopeMultiGrid)
   with Convention => C;  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/driver_types.h:2254

   type cudaLaunchParams is record
      func : System.Address;  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/driver_types.h:2265
      gridDim : aliased vector_types_h.dim3;  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/driver_types.h:2266
      blockDim : aliased vector_types_h.dim3;  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/driver_types.h:2267
      args : System.Address;  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/driver_types.h:2268
      sharedMem : aliased crtdefs_h.size_t;  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/driver_types.h:2269
      stream : cudaStream_t;  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/driver_types.h:2270
   end record
   with Convention => C_Pass_By_Copy;  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/driver_types.h:2263

   type cudaKernelNodeParams is record
      func : System.Address;  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/driver_types.h:2277
      gridDim : aliased vector_types_h.dim3;  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/driver_types.h:2278
      blockDim : aliased vector_types_h.dim3;  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/driver_types.h:2279
      sharedMemBytes : aliased unsigned;  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/driver_types.h:2280
      kernelParams : System.Address;  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/driver_types.h:2281
      extra : System.Address;  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/driver_types.h:2282
   end record
   with Convention => C_Pass_By_Copy;  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/driver_types.h:2276

   type cudaGraphNodeType is 
     (cudaGraphNodeTypeKernel,
      cudaGraphNodeTypeMemcpy,
      cudaGraphNodeTypeMemset,
      cudaGraphNodeTypeHost,
      cudaGraphNodeTypeGraph,
      cudaGraphNodeTypeEmpty,
      cudaGraphNodeTypeCount)
   with Convention => C;  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/driver_types.h:2288

   type CUgraphExec_st is null record;   -- incomplete struct

   type cudaGraphExec_t is access all CUgraphExec_st;  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/driver_types.h:2301

   type cudaGraphExecUpdateResult is 
     (cudaGraphExecUpdateSuccess,
      cudaGraphExecUpdateError,
      cudaGraphExecUpdateErrorTopologyChanged,
      cudaGraphExecUpdateErrorNodeTypeChanged,
      cudaGraphExecUpdateErrorFunctionChanged,
      cudaGraphExecUpdateErrorParametersChanged,
      cudaGraphExecUpdateErrorNotSupported)
   with Convention => C;  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/driver_types.h:2306

end driver_types_h;
