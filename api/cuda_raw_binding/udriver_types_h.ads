pragma Ada_2012;
pragma Style_Checks (Off);
pragma Warnings ("U");

with Interfaces.C; use Interfaces.C;
with System;
with stddef_h;
with Interfaces.C.Extensions;
with uvector_types_h;

package udriver_types_h is

   cudaHostAllocDefault : constant := 16#00#;  --  /usr/local/cuda/include//driver_types.h:81
   cudaHostAllocPortable : constant := 16#01#;  --  /usr/local/cuda/include//driver_types.h:82
   cudaHostAllocMapped : constant := 16#02#;  --  /usr/local/cuda/include//driver_types.h:83
   cudaHostAllocWriteCombined : constant := 16#04#;  --  /usr/local/cuda/include//driver_types.h:84

   cudaHostRegisterDefault : constant := 16#00#;  --  /usr/local/cuda/include//driver_types.h:86
   cudaHostRegisterPortable : constant := 16#01#;  --  /usr/local/cuda/include//driver_types.h:87
   cudaHostRegisterMapped : constant := 16#02#;  --  /usr/local/cuda/include//driver_types.h:88
   cudaHostRegisterIoMemory : constant := 16#04#;  --  /usr/local/cuda/include//driver_types.h:89

   cudaPeerAccessDefault : constant := 16#00#;  --  /usr/local/cuda/include//driver_types.h:91

   cudaStreamDefault : constant := 16#00#;  --  /usr/local/cuda/include//driver_types.h:93
   cudaStreamNonBlocking : constant := 16#01#;  --  /usr/local/cuda/include//driver_types.h:94
   --  unsupported macro: cudaStreamLegacy ((cudaStream_t)0x1)
   --  unsupported macro: cudaStreamPerThread ((cudaStream_t)0x2)

   cudaEventDefault : constant := 16#00#;  --  /usr/local/cuda/include//driver_types.h:116
   cudaEventBlockingSync : constant := 16#01#;  --  /usr/local/cuda/include//driver_types.h:117
   cudaEventDisableTiming : constant := 16#02#;  --  /usr/local/cuda/include//driver_types.h:118
   cudaEventInterprocess : constant := 16#04#;  --  /usr/local/cuda/include//driver_types.h:119

   cudaDeviceScheduleAuto : constant := 16#00#;  --  /usr/local/cuda/include//driver_types.h:121
   cudaDeviceScheduleSpin : constant := 16#01#;  --  /usr/local/cuda/include//driver_types.h:122
   cudaDeviceScheduleYield : constant := 16#02#;  --  /usr/local/cuda/include//driver_types.h:123
   cudaDeviceScheduleBlockingSync : constant := 16#04#;  --  /usr/local/cuda/include//driver_types.h:124
   cudaDeviceBlockingSync : constant := 16#04#;  --  /usr/local/cuda/include//driver_types.h:125

   cudaDeviceScheduleMask : constant := 16#07#;  --  /usr/local/cuda/include//driver_types.h:128
   cudaDeviceMapHost : constant := 16#08#;  --  /usr/local/cuda/include//driver_types.h:129
   cudaDeviceLmemResizeToMax : constant := 16#10#;  --  /usr/local/cuda/include//driver_types.h:130
   cudaDeviceMask : constant := 16#1f#;  --  /usr/local/cuda/include//driver_types.h:131

   cudaArrayDefault : constant := 16#00#;  --  /usr/local/cuda/include//driver_types.h:133
   cudaArrayLayered : constant := 16#01#;  --  /usr/local/cuda/include//driver_types.h:134
   cudaArraySurfaceLoadStore : constant := 16#02#;  --  /usr/local/cuda/include//driver_types.h:135
   cudaArrayCubemap : constant := 16#04#;  --  /usr/local/cuda/include//driver_types.h:136
   cudaArrayTextureGather : constant := 16#08#;  --  /usr/local/cuda/include//driver_types.h:137
   cudaArrayColorAttachment : constant := 16#20#;  --  /usr/local/cuda/include//driver_types.h:138

   cudaIpcMemLazyEnablePeerAccess : constant := 16#01#;  --  /usr/local/cuda/include//driver_types.h:140

   cudaMemAttachGlobal : constant := 16#01#;  --  /usr/local/cuda/include//driver_types.h:142
   cudaMemAttachHost : constant := 16#02#;  --  /usr/local/cuda/include//driver_types.h:143
   cudaMemAttachSingle : constant := 16#04#;  --  /usr/local/cuda/include//driver_types.h:144

   cudaOccupancyDefault : constant := 16#00#;  --  /usr/local/cuda/include//driver_types.h:146
   cudaOccupancyDisableCachingOverride : constant := 16#01#;  --  /usr/local/cuda/include//driver_types.h:147
   --  unsupported macro: cudaCpuDeviceId ((int)-1)
   --  unsupported macro: cudaInvalidDeviceId ((int)-2)

   cudaCooperativeLaunchMultiDeviceNoPreSync : constant := 16#01#;  --  /usr/local/cuda/include//driver_types.h:157

   cudaCooperativeLaunchMultiDeviceNoPostSync : constant := 16#02#;  --  /usr/local/cuda/include//driver_types.h:164
   --  unsupported macro: cudaDevicePropDontCare { {'\0'}, {{0}}, {'\0'}, 0, 0, 0, 0, 0, 0, 0, {0, 0, 0}, {0, 0, 0}, 0, 0, -1, -1, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, {0, 0}, {0, 0}, {0, 0, 0}, {0, 0}, {0, 0, 0}, {0, 0, 0}, 0, {0, 0}, {0, 0, 0}, {0, 0}, 0, {0, 0}, {0, 0, 0}, {0, 0}, {0, 0, 0}, 0, {0, 0}, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, }

   CUDA_IPC_HANDLE_SIZE : constant := 64;  --  /usr/local/cuda/include//driver_types.h:1881

   cudaExternalMemoryDedicated : constant := 16#1#;  --  /usr/local/cuda/include//driver_types.h:1940

   cudaExternalSemaphoreSignalSkipNvSciBufMemSync : constant := 16#01#;  --  /usr/local/cuda/include//driver_types.h:1949

   cudaExternalSemaphoreWaitSkipNvSciBufMemSync : constant := 16#02#;  --  /usr/local/cuda/include//driver_types.h:1958

   cudaNvSciSyncAttrSignal : constant := 16#1#;  --  /usr/local/cuda/include//driver_types.h:1965

   cudaNvSciSyncAttrWait : constant := 16#2#;  --  /usr/local/cuda/include//driver_types.h:1972

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
   cudaErrorApiFailureBase : constant unsigned := 10000;  -- /usr/local/cuda/include//driver_types.h:189

   type cudaChannelFormatKind is 
     (cudaChannelFormatKindSigned,
      cudaChannelFormatKindUnsigned,
      cudaChannelFormatKindFloat,
      cudaChannelFormatKindNone)
   with Convention => C;  -- /usr/local/cuda/include//driver_types.h:973

   type cudaChannelFormatDesc is record
      x : aliased int;  -- /usr/local/cuda/include//driver_types.h:986
      y : aliased int;  -- /usr/local/cuda/include//driver_types.h:987
      z : aliased int;  -- /usr/local/cuda/include//driver_types.h:988
      w : aliased int;  -- /usr/local/cuda/include//driver_types.h:989
      f : aliased cudaChannelFormatKind;  -- /usr/local/cuda/include//driver_types.h:990
   end record
   with Convention => C_Pass_By_Copy;  -- /usr/local/cuda/include//driver_types.h:984

   type cudaArray is null record;   -- incomplete struct

   type cudaArray_t is access all cudaArray;  -- /usr/local/cuda/include//driver_types.h:996

   type cudaArray_const_t is access constant cudaArray;  -- /usr/local/cuda/include//driver_types.h:1001

   type cudaMipmappedArray is null record;   -- incomplete struct

   type cudaMipmappedArray_t is access all cudaMipmappedArray;  -- /usr/local/cuda/include//driver_types.h:1008

   type cudaMipmappedArray_const_t is access constant cudaMipmappedArray;  -- /usr/local/cuda/include//driver_types.h:1013

   type cudaMemoryType is 
     (cudaMemoryTypeUnregistered,
      cudaMemoryTypeHost,
      cudaMemoryTypeDevice,
      cudaMemoryTypeManaged)
   with Convention => C;  -- /usr/local/cuda/include//driver_types.h:1020

   type cudaMemcpyKind is 
     (cudaMemcpyHostToHost,
      cudaMemcpyHostToDevice,
      cudaMemcpyDeviceToHost,
      cudaMemcpyDeviceToDevice,
      cudaMemcpyDefault)
   with Convention => C;  -- /usr/local/cuda/include//driver_types.h:1031

   type cudaPitchedPtr is record
      ptr : System.Address;  -- /usr/local/cuda/include//driver_types.h:1047
      pitch : aliased stddef_h.size_t;  -- /usr/local/cuda/include//driver_types.h:1048
      xsize : aliased stddef_h.size_t;  -- /usr/local/cuda/include//driver_types.h:1049
      ysize : aliased stddef_h.size_t;  -- /usr/local/cuda/include//driver_types.h:1050
   end record
   with Convention => C_Pass_By_Copy;  -- /usr/local/cuda/include//driver_types.h:1045

   type cudaExtent is record
      width : aliased stddef_h.size_t;  -- /usr/local/cuda/include//driver_types.h:1060
      height : aliased stddef_h.size_t;  -- /usr/local/cuda/include//driver_types.h:1061
      depth : aliased stddef_h.size_t;  -- /usr/local/cuda/include//driver_types.h:1062
   end record
   with Convention => C_Pass_By_Copy;  -- /usr/local/cuda/include//driver_types.h:1058

   type cudaPos is record
      x : aliased stddef_h.size_t;  -- /usr/local/cuda/include//driver_types.h:1072
      y : aliased stddef_h.size_t;  -- /usr/local/cuda/include//driver_types.h:1073
      z : aliased stddef_h.size_t;  -- /usr/local/cuda/include//driver_types.h:1074
   end record
   with Convention => C_Pass_By_Copy;  -- /usr/local/cuda/include//driver_types.h:1070

   type cudaMemcpy3DParms is record
      srcArray : cudaArray_t;  -- /usr/local/cuda/include//driver_types.h:1082
      srcPos : aliased cudaPos;  -- /usr/local/cuda/include//driver_types.h:1083
      srcPtr : aliased cudaPitchedPtr;  -- /usr/local/cuda/include//driver_types.h:1084
      dstArray : cudaArray_t;  -- /usr/local/cuda/include//driver_types.h:1086
      dstPos : aliased cudaPos;  -- /usr/local/cuda/include//driver_types.h:1087
      dstPtr : aliased cudaPitchedPtr;  -- /usr/local/cuda/include//driver_types.h:1088
      extent : aliased cudaExtent;  -- /usr/local/cuda/include//driver_types.h:1090
      kind : aliased cudaMemcpyKind;  -- /usr/local/cuda/include//driver_types.h:1091
   end record
   with Convention => C_Pass_By_Copy;  -- /usr/local/cuda/include//driver_types.h:1080

   type cudaMemcpy3DPeerParms is record
      srcArray : cudaArray_t;  -- /usr/local/cuda/include//driver_types.h:1099
      srcPos : aliased cudaPos;  -- /usr/local/cuda/include//driver_types.h:1100
      srcPtr : aliased cudaPitchedPtr;  -- /usr/local/cuda/include//driver_types.h:1101
      srcDevice : aliased int;  -- /usr/local/cuda/include//driver_types.h:1102
      dstArray : cudaArray_t;  -- /usr/local/cuda/include//driver_types.h:1104
      dstPos : aliased cudaPos;  -- /usr/local/cuda/include//driver_types.h:1105
      dstPtr : aliased cudaPitchedPtr;  -- /usr/local/cuda/include//driver_types.h:1106
      dstDevice : aliased int;  -- /usr/local/cuda/include//driver_types.h:1107
      extent : aliased cudaExtent;  -- /usr/local/cuda/include//driver_types.h:1109
   end record
   with Convention => C_Pass_By_Copy;  -- /usr/local/cuda/include//driver_types.h:1097

   type cudaMemsetParams is record
      dst : System.Address;  -- /usr/local/cuda/include//driver_types.h:1116
      pitch : aliased stddef_h.size_t;  -- /usr/local/cuda/include//driver_types.h:1117
      value : aliased unsigned;  -- /usr/local/cuda/include//driver_types.h:1118
      elementSize : aliased unsigned;  -- /usr/local/cuda/include//driver_types.h:1119
      width : aliased stddef_h.size_t;  -- /usr/local/cuda/include//driver_types.h:1120
      height : aliased stddef_h.size_t;  -- /usr/local/cuda/include//driver_types.h:1121
   end record
   with Convention => C_Pass_By_Copy;  -- /usr/local/cuda/include//driver_types.h:1115

   type cudaAccessProperty is 
     (cudaAccessPropertyNormal,
      cudaAccessPropertyStreaming,
      cudaAccessPropertyPersisting)
   with Convention => C;  -- /usr/local/cuda/include//driver_types.h:1127

   type cudaAccessPolicyWindow is record
      base_ptr : System.Address;  -- /usr/local/cuda/include//driver_types.h:1145
      num_bytes : aliased stddef_h.size_t;  -- /usr/local/cuda/include//driver_types.h:1146
      hitRatio : aliased float;  -- /usr/local/cuda/include//driver_types.h:1147
      hitProp : aliased cudaAccessProperty;  -- /usr/local/cuda/include//driver_types.h:1148
      missProp : aliased cudaAccessProperty;  -- /usr/local/cuda/include//driver_types.h:1149
   end record
   with Convention => C_Pass_By_Copy;  -- /usr/local/cuda/include//driver_types.h:1144

   type cudaHostFn_t is access procedure (arg1 : System.Address)
   with Convention => C;  -- /usr/local/cuda/include//driver_types.h:1162

   type cudaHostNodeParams is record
      fn : cudaHostFn_t;  -- /usr/local/cuda/include//driver_types.h:1168
      userData : System.Address;  -- /usr/local/cuda/include//driver_types.h:1169
   end record
   with Convention => C_Pass_By_Copy;  -- /usr/local/cuda/include//driver_types.h:1167

   type cudaStreamCaptureStatus is 
     (cudaStreamCaptureStatusNone,
      cudaStreamCaptureStatusActive,
      cudaStreamCaptureStatusInvalidated)
   with Convention => C;  -- /usr/local/cuda/include//driver_types.h:1175

   type cudaStreamCaptureMode is 
     (cudaStreamCaptureModeGlobal,
      cudaStreamCaptureModeThreadLocal,
      cudaStreamCaptureModeRelaxed)
   with Convention => C;  -- /usr/local/cuda/include//driver_types.h:1186

   subtype cudaSynchronizationPolicy is unsigned;
   cudaSyncPolicyAuto : constant unsigned := 1;
   cudaSyncPolicySpin : constant unsigned := 2;
   cudaSyncPolicyYield : constant unsigned := 3;
   cudaSyncPolicyBlockingSync : constant unsigned := 4;  -- /usr/local/cuda/include//driver_types.h:1192

   subtype cudaStreamAttrID is unsigned;
   cudaStreamAttributeAccessPolicyWindow : constant unsigned := 1;
   cudaStreamAttributeSynchronizationPolicy : constant unsigned := 3;  -- /usr/local/cuda/include//driver_types.h:1202

   type cudaStreamAttrValue (discr : unsigned := 0) is record
      case discr is
         when 0 =>
            accessPolicyWindow : aliased cudaAccessPolicyWindow;  -- /usr/local/cuda/include//driver_types.h:1211
         when others =>
            syncPolicy : aliased cudaSynchronizationPolicy;  -- /usr/local/cuda/include//driver_types.h:1212
      end case;
   end record
   with Convention => C_Pass_By_Copy,
        Unchecked_Union => True;  -- /usr/local/cuda/include//driver_types.h:1210

   type cudaGraphicsResource is null record;   -- incomplete struct

   subtype cudaGraphicsRegisterFlags is unsigned;
   cudaGraphicsRegisterFlagsNone : constant unsigned := 0;
   cudaGraphicsRegisterFlagsReadOnly : constant unsigned := 1;
   cudaGraphicsRegisterFlagsWriteDiscard : constant unsigned := 2;
   cudaGraphicsRegisterFlagsSurfaceLoadStore : constant unsigned := 4;
   cudaGraphicsRegisterFlagsTextureGather : constant unsigned := 8;  -- /usr/local/cuda/include//driver_types.h:1223

   type cudaGraphicsMapFlags is 
     (cudaGraphicsMapFlagsNone,
      cudaGraphicsMapFlagsReadOnly,
      cudaGraphicsMapFlagsWriteDiscard)
   with Convention => C;  -- /usr/local/cuda/include//driver_types.h:1235

   type cudaGraphicsCubeFace is 
     (cudaGraphicsCubeFacePositiveX,
      cudaGraphicsCubeFaceNegativeX,
      cudaGraphicsCubeFacePositiveY,
      cudaGraphicsCubeFaceNegativeY,
      cudaGraphicsCubeFacePositiveZ,
      cudaGraphicsCubeFaceNegativeZ)
   with Convention => C;  -- /usr/local/cuda/include//driver_types.h:1245

   subtype cudaKernelNodeAttrID is unsigned;
   cudaKernelNodeAttributeAccessPolicyWindow : constant unsigned := 1;
   cudaKernelNodeAttributeCooperative : constant unsigned := 2;  -- /usr/local/cuda/include//driver_types.h:1258

   type cudaKernelNodeAttrValue (discr : unsigned := 0) is record
      case discr is
         when 0 =>
            accessPolicyWindow : aliased cudaAccessPolicyWindow;  -- /usr/local/cuda/include//driver_types.h:1267
         when others =>
            cooperative : aliased int;  -- /usr/local/cuda/include//driver_types.h:1268
      end case;
   end record
   with Convention => C_Pass_By_Copy,
        Unchecked_Union => True;  -- /usr/local/cuda/include//driver_types.h:1266

   type cudaResourceType is 
     (cudaResourceTypeArray,
      cudaResourceTypeMipmappedArray,
      cudaResourceTypeLinear,
      cudaResourceTypePitch2D)
   with Convention => C;  -- /usr/local/cuda/include//driver_types.h:1274

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
   with Convention => C;  -- /usr/local/cuda/include//driver_types.h:1285

   type anon936_struct938 is record
      c_array : cudaArray_t;  -- /usr/local/cuda/include//driver_types.h:1332
   end record
   with Convention => C_Pass_By_Copy;
   type anon936_struct939 is record
      mipmap : cudaMipmappedArray_t;  -- /usr/local/cuda/include//driver_types.h:1335
   end record
   with Convention => C_Pass_By_Copy;
   type anon936_struct940 is record
      devPtr : System.Address;  -- /usr/local/cuda/include//driver_types.h:1338
      desc : aliased cudaChannelFormatDesc;  -- /usr/local/cuda/include//driver_types.h:1339
      sizeInBytes : aliased stddef_h.size_t;  -- /usr/local/cuda/include//driver_types.h:1340
   end record
   with Convention => C_Pass_By_Copy;
   type anon936_struct941 is record
      devPtr : System.Address;  -- /usr/local/cuda/include//driver_types.h:1343
      desc : aliased cudaChannelFormatDesc;  -- /usr/local/cuda/include//driver_types.h:1344
      width : aliased stddef_h.size_t;  -- /usr/local/cuda/include//driver_types.h:1345
      height : aliased stddef_h.size_t;  -- /usr/local/cuda/include//driver_types.h:1346
      pitchInBytes : aliased stddef_h.size_t;  -- /usr/local/cuda/include//driver_types.h:1347
   end record
   with Convention => C_Pass_By_Copy;
   type anon936_union937 (discr : unsigned := 0) is record
      case discr is
         when 0 =>
            c_array : aliased anon936_struct938;  -- /usr/local/cuda/include//driver_types.h:1333
         when 1 =>
            mipmap : aliased anon936_struct939;  -- /usr/local/cuda/include//driver_types.h:1336
         when 2 =>
            linear : aliased anon936_struct940;  -- /usr/local/cuda/include//driver_types.h:1341
         when others =>
            pitch2D : aliased anon936_struct941;  -- /usr/local/cuda/include//driver_types.h:1348
      end case;
   end record
   with Convention => C_Pass_By_Copy,
        Unchecked_Union => True;
   type cudaResourceDesc is record
      resType : aliased cudaResourceType;  -- /usr/local/cuda/include//driver_types.h:1328
      res : aliased anon936_union937;  -- /usr/local/cuda/include//driver_types.h:1349
   end record
   with Convention => C_Pass_By_Copy;  -- /usr/local/cuda/include//driver_types.h:1327

   type cudaResourceViewDesc is record
      format : aliased cudaResourceViewFormat;  -- /usr/local/cuda/include//driver_types.h:1357
      width : aliased stddef_h.size_t;  -- /usr/local/cuda/include//driver_types.h:1358
      height : aliased stddef_h.size_t;  -- /usr/local/cuda/include//driver_types.h:1359
      depth : aliased stddef_h.size_t;  -- /usr/local/cuda/include//driver_types.h:1360
      firstMipmapLevel : aliased unsigned;  -- /usr/local/cuda/include//driver_types.h:1361
      lastMipmapLevel : aliased unsigned;  -- /usr/local/cuda/include//driver_types.h:1362
      firstLayer : aliased unsigned;  -- /usr/local/cuda/include//driver_types.h:1363
      lastLayer : aliased unsigned;  -- /usr/local/cuda/include//driver_types.h:1364
   end record
   with Convention => C_Pass_By_Copy;  -- /usr/local/cuda/include//driver_types.h:1355

   type cudaPointerAttributes is record
      c_type : aliased cudaMemoryType;  -- /usr/local/cuda/include//driver_types.h:1376
      device : aliased int;  -- /usr/local/cuda/include//driver_types.h:1387
      devicePointer : System.Address;  -- /usr/local/cuda/include//driver_types.h:1393
      hostPointer : System.Address;  -- /usr/local/cuda/include//driver_types.h:1402
   end record
   with Convention => C_Pass_By_Copy;  -- /usr/local/cuda/include//driver_types.h:1370

   type cudaFuncAttributes is record
      sharedSizeBytes : aliased stddef_h.size_t;  -- /usr/local/cuda/include//driver_types.h:1415
      constSizeBytes : aliased stddef_h.size_t;  -- /usr/local/cuda/include//driver_types.h:1421
      localSizeBytes : aliased stddef_h.size_t;  -- /usr/local/cuda/include//driver_types.h:1426
      maxThreadsPerBlock : aliased int;  -- /usr/local/cuda/include//driver_types.h:1433
      numRegs : aliased int;  -- /usr/local/cuda/include//driver_types.h:1438
      ptxVersion : aliased int;  -- /usr/local/cuda/include//driver_types.h:1445
      binaryVersion : aliased int;  -- /usr/local/cuda/include//driver_types.h:1452
      cacheModeCA : aliased int;  -- /usr/local/cuda/include//driver_types.h:1458
      maxDynamicSharedSizeBytes : aliased int;  -- /usr/local/cuda/include//driver_types.h:1465
      preferredShmemCarveout : aliased int;  -- /usr/local/cuda/include//driver_types.h:1474
   end record
   with Convention => C_Pass_By_Copy;  -- /usr/local/cuda/include//driver_types.h:1408

   subtype cudaFuncAttribute is unsigned;
   cudaFuncAttributeMaxDynamicSharedMemorySize : constant unsigned := 8;
   cudaFuncAttributePreferredSharedMemoryCarveout : constant unsigned := 9;
   cudaFuncAttributeMax : constant unsigned := 10;  -- /usr/local/cuda/include//driver_types.h:1480

   type cudaFuncCache is 
     (cudaFuncCachePreferNone,
      cudaFuncCachePreferShared,
      cudaFuncCachePreferL1,
      cudaFuncCachePreferEqual)
   with Convention => C;  -- /usr/local/cuda/include//driver_types.h:1490

   type cudaSharedMemConfig is 
     (cudaSharedMemBankSizeDefault,
      cudaSharedMemBankSizeFourByte,
      cudaSharedMemBankSizeEightByte)
   with Convention => C;  -- /usr/local/cuda/include//driver_types.h:1502

   subtype cudaSharedCarveout is int;
   cudaSharedmemCarveoutDefault : constant int := -1;
   cudaSharedmemCarveoutMaxShared : constant int := 100;
   cudaSharedmemCarveoutMaxL1 : constant int := 0;  -- /usr/local/cuda/include//driver_types.h:1512

   type cudaComputeMode is 
     (cudaComputeModeDefault,
      cudaComputeModeExclusive,
      cudaComputeModeProhibited,
      cudaComputeModeExclusiveProcess)
   with Convention => C;  -- /usr/local/cuda/include//driver_types.h:1521

   type cudaLimit is 
     (cudaLimitStackSize,
      cudaLimitPrintfFifoSize,
      cudaLimitMallocHeapSize,
      cudaLimitDevRuntimeSyncDepth,
      cudaLimitDevRuntimePendingLaunchCount,
      cudaLimitMaxL2FetchGranularity,
      cudaLimitPersistingL2CacheSize)
   with Convention => C;  -- /usr/local/cuda/include//driver_types.h:1532

   subtype cudaMemoryAdvise is unsigned;
   cudaMemAdviseSetReadMostly : constant unsigned := 1;
   cudaMemAdviseUnsetReadMostly : constant unsigned := 2;
   cudaMemAdviseSetPreferredLocation : constant unsigned := 3;
   cudaMemAdviseUnsetPreferredLocation : constant unsigned := 4;
   cudaMemAdviseSetAccessedBy : constant unsigned := 5;
   cudaMemAdviseUnsetAccessedBy : constant unsigned := 6;  -- /usr/local/cuda/include//driver_types.h:1546

   subtype cudaMemRangeAttribute is unsigned;
   cudaMemRangeAttributeReadMostly : constant unsigned := 1;
   cudaMemRangeAttributePreferredLocation : constant unsigned := 2;
   cudaMemRangeAttributeAccessedBy : constant unsigned := 3;
   cudaMemRangeAttributeLastPrefetchLocation : constant unsigned := 4;  -- /usr/local/cuda/include//driver_types.h:1559

   type cudaOutputMode is 
     (cudaKeyValuePair,
      cudaCSV)
   with Convention => C;  -- /usr/local/cuda/include//driver_types.h:1570

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
   cudaDevAttrDirectManagedMemAccessFromHost : constant unsigned := 101;
   cudaDevAttrMaxBlocksPerMultiprocessor : constant unsigned := 106;
   cudaDevAttrReservedSharedMemoryPerBlock : constant unsigned := 111;  -- /usr/local/cuda/include//driver_types.h:1579

   subtype cudaDeviceP2PAttr is unsigned;
   cudaDevP2PAttrPerformanceRank : constant unsigned := 1;
   cudaDevP2PAttrAccessSupported : constant unsigned := 2;
   cudaDevP2PAttrNativeAtomicSupported : constant unsigned := 3;
   cudaDevP2PAttrCudaArrayAccessSupported : constant unsigned := 4;  -- /usr/local/cuda/include//driver_types.h:1689

   subtype anon956_array958 is Interfaces.C.char_array (0 .. 15);
   type CUuuid_st is record
      bytes : aliased anon956_array958;  -- /usr/local/cuda/include//driver_types.h:1702
   end record
   with Convention => C_Pass_By_Copy;  -- /usr/local/cuda/include//driver_types.h:1701

   subtype CUuuid is CUuuid_st;  -- /usr/local/cuda/include//driver_types.h:1704

   subtype cudaUUID_t is CUuuid_st;  -- /usr/local/cuda/include//driver_types.h:1706

   subtype anon961_array963 is Interfaces.C.char_array (0 .. 255);
   subtype anon961_array965 is Interfaces.C.char_array (0 .. 7);
   type anon961_array967 is array (0 .. 2) of aliased int;
   type anon961_array969 is array (0 .. 1) of aliased int;
   type cudaDeviceProp is record
      name : aliased anon961_array963;  -- /usr/local/cuda/include//driver_types.h:1713
      uuid : aliased cudaUUID_t;  -- /usr/local/cuda/include//driver_types.h:1714
      luid : aliased anon961_array965;  -- /usr/local/cuda/include//driver_types.h:1715
      luidDeviceNodeMask : aliased unsigned;  -- /usr/local/cuda/include//driver_types.h:1716
      totalGlobalMem : aliased stddef_h.size_t;  -- /usr/local/cuda/include//driver_types.h:1717
      sharedMemPerBlock : aliased stddef_h.size_t;  -- /usr/local/cuda/include//driver_types.h:1718
      regsPerBlock : aliased int;  -- /usr/local/cuda/include//driver_types.h:1719
      warpSize : aliased int;  -- /usr/local/cuda/include//driver_types.h:1720
      memPitch : aliased stddef_h.size_t;  -- /usr/local/cuda/include//driver_types.h:1721
      maxThreadsPerBlock : aliased int;  -- /usr/local/cuda/include//driver_types.h:1722
      maxThreadsDim : aliased anon961_array967;  -- /usr/local/cuda/include//driver_types.h:1723
      maxGridSize : aliased anon961_array967;  -- /usr/local/cuda/include//driver_types.h:1724
      clockRate : aliased int;  -- /usr/local/cuda/include//driver_types.h:1725
      totalConstMem : aliased stddef_h.size_t;  -- /usr/local/cuda/include//driver_types.h:1726
      major : aliased int;  -- /usr/local/cuda/include//driver_types.h:1727
      minor : aliased int;  -- /usr/local/cuda/include//driver_types.h:1728
      textureAlignment : aliased stddef_h.size_t;  -- /usr/local/cuda/include//driver_types.h:1729
      texturePitchAlignment : aliased stddef_h.size_t;  -- /usr/local/cuda/include//driver_types.h:1730
      deviceOverlap : aliased int;  -- /usr/local/cuda/include//driver_types.h:1731
      multiProcessorCount : aliased int;  -- /usr/local/cuda/include//driver_types.h:1732
      kernelExecTimeoutEnabled : aliased int;  -- /usr/local/cuda/include//driver_types.h:1733
      integrated : aliased int;  -- /usr/local/cuda/include//driver_types.h:1734
      canMapHostMemory : aliased int;  -- /usr/local/cuda/include//driver_types.h:1735
      computeMode : aliased int;  -- /usr/local/cuda/include//driver_types.h:1736
      maxTexture1D : aliased int;  -- /usr/local/cuda/include//driver_types.h:1737
      maxTexture1DMipmap : aliased int;  -- /usr/local/cuda/include//driver_types.h:1738
      maxTexture1DLinear : aliased int;  -- /usr/local/cuda/include//driver_types.h:1739
      maxTexture2D : aliased anon961_array969;  -- /usr/local/cuda/include//driver_types.h:1740
      maxTexture2DMipmap : aliased anon961_array969;  -- /usr/local/cuda/include//driver_types.h:1741
      maxTexture2DLinear : aliased anon961_array967;  -- /usr/local/cuda/include//driver_types.h:1742
      maxTexture2DGather : aliased anon961_array969;  -- /usr/local/cuda/include//driver_types.h:1743
      maxTexture3D : aliased anon961_array967;  -- /usr/local/cuda/include//driver_types.h:1744
      maxTexture3DAlt : aliased anon961_array967;  -- /usr/local/cuda/include//driver_types.h:1745
      maxTextureCubemap : aliased int;  -- /usr/local/cuda/include//driver_types.h:1746
      maxTexture1DLayered : aliased anon961_array969;  -- /usr/local/cuda/include//driver_types.h:1747
      maxTexture2DLayered : aliased anon961_array967;  -- /usr/local/cuda/include//driver_types.h:1748
      maxTextureCubemapLayered : aliased anon961_array969;  -- /usr/local/cuda/include//driver_types.h:1749
      maxSurface1D : aliased int;  -- /usr/local/cuda/include//driver_types.h:1750
      maxSurface2D : aliased anon961_array969;  -- /usr/local/cuda/include//driver_types.h:1751
      maxSurface3D : aliased anon961_array967;  -- /usr/local/cuda/include//driver_types.h:1752
      maxSurface1DLayered : aliased anon961_array969;  -- /usr/local/cuda/include//driver_types.h:1753
      maxSurface2DLayered : aliased anon961_array967;  -- /usr/local/cuda/include//driver_types.h:1754
      maxSurfaceCubemap : aliased int;  -- /usr/local/cuda/include//driver_types.h:1755
      maxSurfaceCubemapLayered : aliased anon961_array969;  -- /usr/local/cuda/include//driver_types.h:1756
      surfaceAlignment : aliased stddef_h.size_t;  -- /usr/local/cuda/include//driver_types.h:1757
      concurrentKernels : aliased int;  -- /usr/local/cuda/include//driver_types.h:1758
      ECCEnabled : aliased int;  -- /usr/local/cuda/include//driver_types.h:1759
      pciBusID : aliased int;  -- /usr/local/cuda/include//driver_types.h:1760
      pciDeviceID : aliased int;  -- /usr/local/cuda/include//driver_types.h:1761
      pciDomainID : aliased int;  -- /usr/local/cuda/include//driver_types.h:1762
      tccDriver : aliased int;  -- /usr/local/cuda/include//driver_types.h:1763
      asyncEngineCount : aliased int;  -- /usr/local/cuda/include//driver_types.h:1764
      unifiedAddressing : aliased int;  -- /usr/local/cuda/include//driver_types.h:1765
      memoryClockRate : aliased int;  -- /usr/local/cuda/include//driver_types.h:1766
      memoryBusWidth : aliased int;  -- /usr/local/cuda/include//driver_types.h:1767
      l2CacheSize : aliased int;  -- /usr/local/cuda/include//driver_types.h:1768
      persistingL2CacheMaxSize : aliased int;  -- /usr/local/cuda/include//driver_types.h:1769
      maxThreadsPerMultiProcessor : aliased int;  -- /usr/local/cuda/include//driver_types.h:1770
      streamPrioritiesSupported : aliased int;  -- /usr/local/cuda/include//driver_types.h:1771
      globalL1CacheSupported : aliased int;  -- /usr/local/cuda/include//driver_types.h:1772
      localL1CacheSupported : aliased int;  -- /usr/local/cuda/include//driver_types.h:1773
      sharedMemPerMultiprocessor : aliased stddef_h.size_t;  -- /usr/local/cuda/include//driver_types.h:1774
      regsPerMultiprocessor : aliased int;  -- /usr/local/cuda/include//driver_types.h:1775
      managedMemory : aliased int;  -- /usr/local/cuda/include//driver_types.h:1776
      isMultiGpuBoard : aliased int;  -- /usr/local/cuda/include//driver_types.h:1777
      multiGpuBoardGroupID : aliased int;  -- /usr/local/cuda/include//driver_types.h:1778
      hostNativeAtomicSupported : aliased int;  -- /usr/local/cuda/include//driver_types.h:1779
      singleToDoublePrecisionPerfRatio : aliased int;  -- /usr/local/cuda/include//driver_types.h:1780
      pageableMemoryAccess : aliased int;  -- /usr/local/cuda/include//driver_types.h:1781
      concurrentManagedAccess : aliased int;  -- /usr/local/cuda/include//driver_types.h:1782
      computePreemptionSupported : aliased int;  -- /usr/local/cuda/include//driver_types.h:1783
      canUseHostPointerForRegisteredMem : aliased int;  -- /usr/local/cuda/include//driver_types.h:1784
      cooperativeLaunch : aliased int;  -- /usr/local/cuda/include//driver_types.h:1785
      cooperativeMultiDeviceLaunch : aliased int;  -- /usr/local/cuda/include//driver_types.h:1786
      sharedMemPerBlockOptin : aliased stddef_h.size_t;  -- /usr/local/cuda/include//driver_types.h:1787
      pageableMemoryAccessUsesHostPageTables : aliased int;  -- /usr/local/cuda/include//driver_types.h:1788
      directManagedMemAccessFromHost : aliased int;  -- /usr/local/cuda/include//driver_types.h:1789
      maxBlocksPerMultiProcessor : aliased int;  -- /usr/local/cuda/include//driver_types.h:1790
      accessPolicyMaxWindowSize : aliased int;  -- /usr/local/cuda/include//driver_types.h:1791
      reservedSharedMemPerBlock : aliased stddef_h.size_t;  -- /usr/local/cuda/include//driver_types.h:1792
   end record
   with Convention => C_Pass_By_Copy;  -- /usr/local/cuda/include//driver_types.h:1711

   subtype anon970_array972 is Interfaces.C.char_array (0 .. 63);
   type cudaIpcEventHandle_st is record
      reserved : aliased anon970_array972;  -- /usr/local/cuda/include//driver_types.h:1888
   end record
   with Convention => C_Pass_By_Copy;  -- /usr/local/cuda/include//driver_types.h:1886

   subtype cudaIpcEventHandle_t is cudaIpcEventHandle_st;  -- /usr/local/cuda/include//driver_types.h:1889

   subtype anon974_array972 is Interfaces.C.char_array (0 .. 63);
   type cudaIpcMemHandle_st is record
      reserved : aliased anon974_array972;  -- /usr/local/cuda/include//driver_types.h:1896
   end record
   with Convention => C_Pass_By_Copy;  -- /usr/local/cuda/include//driver_types.h:1894

   subtype cudaIpcMemHandle_t is cudaIpcMemHandle_st;  -- /usr/local/cuda/include//driver_types.h:1897

   subtype cudaExternalMemoryHandleType is unsigned;
   cudaExternalMemoryHandleTypeOpaqueFd : constant unsigned := 1;
   cudaExternalMemoryHandleTypeOpaqueWin32 : constant unsigned := 2;
   cudaExternalMemoryHandleTypeOpaqueWin32Kmt : constant unsigned := 3;
   cudaExternalMemoryHandleTypeD3D12Heap : constant unsigned := 4;
   cudaExternalMemoryHandleTypeD3D12Resource : constant unsigned := 5;
   cudaExternalMemoryHandleTypeD3D11Resource : constant unsigned := 6;
   cudaExternalMemoryHandleTypeD3D11ResourceKmt : constant unsigned := 7;
   cudaExternalMemoryHandleTypeNvSciBuf : constant unsigned := 8;  -- /usr/local/cuda/include//driver_types.h:1902

   type anon977_struct979 is record
      handle : System.Address;  -- /usr/local/cuda/include//driver_types.h:2008
      name : System.Address;  -- /usr/local/cuda/include//driver_types.h:2013
   end record
   with Convention => C_Pass_By_Copy;
   type anon977_union978 (discr : unsigned := 0) is record
      case discr is
         when 0 =>
            fd : aliased int;  -- /usr/local/cuda/include//driver_types.h:1988
         when 1 =>
            win32 : aliased anon977_struct979;  -- /usr/local/cuda/include//driver_types.h:2014
         when others =>
            nvSciBufObject : System.Address;  -- /usr/local/cuda/include//driver_types.h:2019
      end case;
   end record
   with Convention => C_Pass_By_Copy,
        Unchecked_Union => True;
   type cudaExternalMemoryHandleDesc is record
      c_type : aliased cudaExternalMemoryHandleType;  -- /usr/local/cuda/include//driver_types.h:1981
      handle : aliased anon977_union978;  -- /usr/local/cuda/include//driver_types.h:2020
      size : aliased Extensions.unsigned_long_long;  -- /usr/local/cuda/include//driver_types.h:2024
      flags : aliased unsigned;  -- /usr/local/cuda/include//driver_types.h:2028
   end record
   with Convention => C_Pass_By_Copy;  -- /usr/local/cuda/include//driver_types.h:1977

   type cudaExternalMemoryBufferDesc is record
      offset : aliased Extensions.unsigned_long_long;  -- /usr/local/cuda/include//driver_types.h:2038
      size : aliased Extensions.unsigned_long_long;  -- /usr/local/cuda/include//driver_types.h:2042
      flags : aliased unsigned;  -- /usr/local/cuda/include//driver_types.h:2046
   end record
   with Convention => C_Pass_By_Copy;  -- /usr/local/cuda/include//driver_types.h:2034

   type cudaExternalMemoryMipmappedArrayDesc is record
      offset : aliased Extensions.unsigned_long_long;  -- /usr/local/cuda/include//driver_types.h:2057
      formatDesc : aliased cudaChannelFormatDesc;  -- /usr/local/cuda/include//driver_types.h:2061
      extent : aliased cudaExtent;  -- /usr/local/cuda/include//driver_types.h:2065
      flags : aliased unsigned;  -- /usr/local/cuda/include//driver_types.h:2070
      numLevels : aliased unsigned;  -- /usr/local/cuda/include//driver_types.h:2074
   end record
   with Convention => C_Pass_By_Copy;  -- /usr/local/cuda/include//driver_types.h:2052

   subtype cudaExternalSemaphoreHandleType is unsigned;
   cudaExternalSemaphoreHandleTypeOpaqueFd : constant unsigned := 1;
   cudaExternalSemaphoreHandleTypeOpaqueWin32 : constant unsigned := 2;
   cudaExternalSemaphoreHandleTypeOpaqueWin32Kmt : constant unsigned := 3;
   cudaExternalSemaphoreHandleTypeD3D12Fence : constant unsigned := 4;
   cudaExternalSemaphoreHandleTypeD3D11Fence : constant unsigned := 5;
   cudaExternalSemaphoreHandleTypeNvSciSync : constant unsigned := 6;
   cudaExternalSemaphoreHandleTypeKeyedMutex : constant unsigned := 7;
   cudaExternalSemaphoreHandleTypeKeyedMutexKmt : constant unsigned := 8;  -- /usr/local/cuda/include//driver_types.h:2080

   type anon983_struct985 is record
      handle : System.Address;  -- /usr/local/cuda/include//driver_types.h:2147
      name : System.Address;  -- /usr/local/cuda/include//driver_types.h:2152
   end record
   with Convention => C_Pass_By_Copy;
   type anon983_union984 (discr : unsigned := 0) is record
      case discr is
         when 0 =>
            fd : aliased int;  -- /usr/local/cuda/include//driver_types.h:2128
         when 1 =>
            win32 : aliased anon983_struct985;  -- /usr/local/cuda/include//driver_types.h:2153
         when others =>
            nvSciSyncObj : System.Address;  -- /usr/local/cuda/include//driver_types.h:2157
      end case;
   end record
   with Convention => C_Pass_By_Copy,
        Unchecked_Union => True;
   type cudaExternalSemaphoreHandleDesc is record
      c_type : aliased cudaExternalSemaphoreHandleType;  -- /usr/local/cuda/include//driver_types.h:2122
      handle : aliased anon983_union984;  -- /usr/local/cuda/include//driver_types.h:2158
      flags : aliased unsigned;  -- /usr/local/cuda/include//driver_types.h:2162
   end record
   with Convention => C_Pass_By_Copy;  -- /usr/local/cuda/include//driver_types.h:2118

   type anon986_struct988 is record
      value : aliased Extensions.unsigned_long_long;  -- /usr/local/cuda/include//driver_types.h:2177
   end record
   with Convention => C_Pass_By_Copy;
   type anon986_union989 (discr : unsigned := 0) is record
      case discr is
         when 0 =>
            fence : System.Address;  -- /usr/local/cuda/include//driver_types.h:2184
         when others =>
            reserved : aliased Extensions.unsigned_long_long;  -- /usr/local/cuda/include//driver_types.h:2185
      end case;
   end record
   with Convention => C_Pass_By_Copy,
        Unchecked_Union => True;
   type anon986_struct990 is record
      key : aliased Extensions.unsigned_long_long;  -- /usr/local/cuda/include//driver_types.h:2194
   end record
   with Convention => C_Pass_By_Copy;
   type anon986_struct987 is record
      fence : aliased anon986_struct988;  -- /usr/local/cuda/include//driver_types.h:2178
      nvSciSync : aliased anon986_union989;  -- /usr/local/cuda/include//driver_types.h:2186
      keyedMutex : aliased anon986_struct990;  -- /usr/local/cuda/include//driver_types.h:2195
   end record
   with Convention => C_Pass_By_Copy;
   type cudaExternalSemaphoreSignalParams is record
      params : aliased anon986_struct987;  -- /usr/local/cuda/include//driver_types.h:2196
      flags : aliased unsigned;  -- /usr/local/cuda/include//driver_types.h:2207
   end record
   with Convention => C_Pass_By_Copy;  -- /usr/local/cuda/include//driver_types.h:2168

   type anon991_struct993 is record
      value : aliased Extensions.unsigned_long_long;  -- /usr/local/cuda/include//driver_types.h:2222
   end record
   with Convention => C_Pass_By_Copy;
   type anon991_union994 (discr : unsigned := 0) is record
      case discr is
         when 0 =>
            fence : System.Address;  -- /usr/local/cuda/include//driver_types.h:2229
         when others =>
            reserved : aliased Extensions.unsigned_long_long;  -- /usr/local/cuda/include//driver_types.h:2230
      end case;
   end record
   with Convention => C_Pass_By_Copy,
        Unchecked_Union => True;
   type anon991_struct995 is record
      key : aliased Extensions.unsigned_long_long;  -- /usr/local/cuda/include//driver_types.h:2239
      timeoutMs : aliased unsigned;  -- /usr/local/cuda/include//driver_types.h:2243
   end record
   with Convention => C_Pass_By_Copy;
   type anon991_struct992 is record
      fence : aliased anon991_struct993;  -- /usr/local/cuda/include//driver_types.h:2223
      nvSciSync : aliased anon991_union994;  -- /usr/local/cuda/include//driver_types.h:2231
      keyedMutex : aliased anon991_struct995;  -- /usr/local/cuda/include//driver_types.h:2244
   end record
   with Convention => C_Pass_By_Copy;
   type cudaExternalSemaphoreWaitParams is record
      params : aliased anon991_struct992;  -- /usr/local/cuda/include//driver_types.h:2245
      flags : aliased unsigned;  -- /usr/local/cuda/include//driver_types.h:2256
   end record
   with Convention => C_Pass_By_Copy;  -- /usr/local/cuda/include//driver_types.h:2213

   subtype cudaError_t is cudaError;  -- /usr/local/cuda/include//driver_types.h:2269

   type CUstream_st is null record;   -- incomplete struct

   type cudaStream_t is access all CUstream_st;  -- /usr/local/cuda/include//driver_types.h:2274

   type CUevent_st is null record;   -- incomplete struct

   type cudaEvent_t is access all CUevent_st;  -- /usr/local/cuda/include//driver_types.h:2279

   type cudaGraphicsResource_t is access all cudaGraphicsResource;  -- /usr/local/cuda/include//driver_types.h:2284

   subtype cudaOutputMode_t is cudaOutputMode;  -- /usr/local/cuda/include//driver_types.h:2289

   type CUexternalMemory_st is null record;   -- incomplete struct

   type cudaExternalMemory_t is access all CUexternalMemory_st;  -- /usr/local/cuda/include//driver_types.h:2294

   type CUexternalSemaphore_st is null record;   -- incomplete struct

   type cudaExternalSemaphore_t is access all CUexternalSemaphore_st;  -- /usr/local/cuda/include//driver_types.h:2299

   type CUgraph_st is null record;   -- incomplete struct

   type cudaGraph_t is access all CUgraph_st;  -- /usr/local/cuda/include//driver_types.h:2304

   type CUgraphNode_st is null record;   -- incomplete struct

   type cudaGraphNode_t is access all CUgraphNode_st;  -- /usr/local/cuda/include//driver_types.h:2309

   type CUfunc_st is null record;   -- incomplete struct

   type cudaFunction_t is access all CUfunc_st;  -- /usr/local/cuda/include//driver_types.h:2314

   type cudaCGScope is 
     (cudaCGScopeInvalid,
      cudaCGScopeGrid,
      cudaCGScopeMultiGrid)
   with Convention => C;  -- /usr/local/cuda/include//driver_types.h:2319

   type cudaLaunchParams is record
      func : System.Address;  -- /usr/local/cuda/include//driver_types.h:2330
      gridDim : aliased uvector_types_h.dim3;  -- /usr/local/cuda/include//driver_types.h:2331
      blockDim : aliased uvector_types_h.dim3;  -- /usr/local/cuda/include//driver_types.h:2332
      args : System.Address;  -- /usr/local/cuda/include//driver_types.h:2333
      sharedMem : aliased stddef_h.size_t;  -- /usr/local/cuda/include//driver_types.h:2334
      stream : cudaStream_t;  -- /usr/local/cuda/include//driver_types.h:2335
   end record
   with Convention => C_Pass_By_Copy;  -- /usr/local/cuda/include//driver_types.h:2328

   type cudaKernelNodeParams is record
      func : System.Address;  -- /usr/local/cuda/include//driver_types.h:2342
      gridDim : aliased uvector_types_h.dim3;  -- /usr/local/cuda/include//driver_types.h:2343
      blockDim : aliased uvector_types_h.dim3;  -- /usr/local/cuda/include//driver_types.h:2344
      sharedMemBytes : aliased unsigned;  -- /usr/local/cuda/include//driver_types.h:2345
      kernelParams : System.Address;  -- /usr/local/cuda/include//driver_types.h:2346
      extra : System.Address;  -- /usr/local/cuda/include//driver_types.h:2347
   end record
   with Convention => C_Pass_By_Copy;  -- /usr/local/cuda/include//driver_types.h:2341

   type cudaGraphNodeType is 
     (cudaGraphNodeTypeKernel,
      cudaGraphNodeTypeMemcpy,
      cudaGraphNodeTypeMemset,
      cudaGraphNodeTypeHost,
      cudaGraphNodeTypeGraph,
      cudaGraphNodeTypeEmpty,
      cudaGraphNodeTypeCount)
   with Convention => C;  -- /usr/local/cuda/include//driver_types.h:2353

   type CUgraphExec_st is null record;   -- incomplete struct

   type cudaGraphExec_t is access all CUgraphExec_st;  -- /usr/local/cuda/include//driver_types.h:2366

   type cudaGraphExecUpdateResult is 
     (cudaGraphExecUpdateSuccess,
      cudaGraphExecUpdateError,
      cudaGraphExecUpdateErrorTopologyChanged,
      cudaGraphExecUpdateErrorNodeTypeChanged,
      cudaGraphExecUpdateErrorFunctionChanged,
      cudaGraphExecUpdateErrorParametersChanged,
      cudaGraphExecUpdateErrorNotSupported)
   with Convention => C;  -- /usr/local/cuda/include//driver_types.h:2371

end udriver_types_h;
