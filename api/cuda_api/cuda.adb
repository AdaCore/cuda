package body CUDA is

begin
   null;
   Exception_Registry.Insert (Integer (1), ErrorInvalidValue'Identity);
   Exception_Registry.Insert (Integer (2), ErrorMemoryAllocation'Identity);
   Exception_Registry.Insert (Integer (3), ErrorInitializationError'Identity);
   Exception_Registry.Insert (Integer (4), ErrorCudartUnloading'Identity);
   Exception_Registry.Insert (Integer (5), ErrorProfilerDisabled'Identity);
   Exception_Registry.Insert
     (Integer (6), ErrorProfilerNotInitialized'Identity);
   Exception_Registry.Insert
     (Integer (7), ErrorProfilerAlreadyStarted'Identity);
   Exception_Registry.Insert
     (Integer (8), ErrorProfilerAlreadyStopped'Identity);
   Exception_Registry.Insert (Integer (9), ErrorInvalidConfiguration'Identity);
   Exception_Registry.Insert (Integer (12), ErrorInvalidPitchValue'Identity);
   Exception_Registry.Insert (Integer (13), ErrorInvalidSymbol'Identity);
   Exception_Registry.Insert (Integer (16), ErrorInvalidHostPointer'Identity);
   Exception_Registry.Insert (Integer (17), ErrorInvalidDevicePointer'Identity);
   Exception_Registry.Insert (Integer (18), ErrorInvalidTexture'Identity);
   Exception_Registry.Insert
     (Integer (19), ErrorInvalidTextureBinding'Identity);
   Exception_Registry.Insert
     (Integer (20), ErrorInvalidChannelDescriptor'Identity);
   Exception_Registry.Insert
     (Integer (21), ErrorInvalidMemcpyDirection'Identity);
   Exception_Registry.Insert (Integer (22), ErrorAddressOfConstant'Identity);
   Exception_Registry.Insert (Integer (23), ErrorTextureFetchFailed'Identity);
   Exception_Registry.Insert (Integer (24), ErrorTextureNotBound'Identity);
   Exception_Registry.Insert (Integer (25), ErrorSynchronizationError'Identity);
   Exception_Registry.Insert (Integer (26), ErrorInvalidFilterSetting'Identity);
   Exception_Registry.Insert (Integer (27), ErrorInvalidNormSetting'Identity);
   Exception_Registry.Insert (Integer (28), ErrorMixedDeviceExecution'Identity);
   Exception_Registry.Insert (Integer (31), ErrorNotYetImplemented'Identity);
   Exception_Registry.Insert (Integer (32), ErrorMemoryValueTooLarge'Identity);
   Exception_Registry.Insert (Integer (35), ErrorInsufficientDriver'Identity);
   Exception_Registry.Insert (Integer (37), ErrorInvalidSurface'Identity);
   Exception_Registry.Insert
     (Integer (43), ErrorDuplicateVariableName'Identity);
   Exception_Registry.Insert (Integer (44), ErrorDuplicateTextureName'Identity);
   Exception_Registry.Insert (Integer (45), ErrorDuplicateSurfaceName'Identity);
   Exception_Registry.Insert (Integer (46), ErrorDevicesUnavailable'Identity);
   Exception_Registry.Insert
     (Integer (49), ErrorIncompatibleDriverContext'Identity);
   Exception_Registry.Insert (Integer (52), ErrorMissingConfiguration'Identity);
   Exception_Registry.Insert (Integer (53), ErrorPriorLaunchFailure'Identity);
   Exception_Registry.Insert
     (Integer (65), ErrorLaunchMaxDepthExceeded'Identity);
   Exception_Registry.Insert (Integer (66), ErrorLaunchFileScopedTex'Identity);
   Exception_Registry.Insert (Integer (67), ErrorLaunchFileScopedSurf'Identity);
   Exception_Registry.Insert (Integer (68), ErrorSyncDepthExceeded'Identity);
   Exception_Registry.Insert
     (Integer (69), ErrorLaunchPendingCountExceeded'Identity);
   Exception_Registry.Insert
     (Integer (98), ErrorInvalidDeviceFunction'Identity);
   Exception_Registry.Insert (Integer (100), ErrorNoDevice'Identity);
   Exception_Registry.Insert (Integer (101), ErrorInvalidDevice'Identity);
   Exception_Registry.Insert (Integer (127), ErrorStartupFailure'Identity);
   Exception_Registry.Insert (Integer (200), ErrorInvalidKernelImage'Identity);
   Exception_Registry.Insert (Integer (201), ErrorDeviceUninitialized'Identity);
   Exception_Registry.Insert
     (Integer (205), ErrorMapBufferObjectFailed'Identity);
   Exception_Registry.Insert
     (Integer (206), ErrorUnmapBufferObjectFailed'Identity);
   Exception_Registry.Insert (Integer (207), ErrorArrayIsMapped'Identity);
   Exception_Registry.Insert (Integer (208), ErrorAlreadyMapped'Identity);
   Exception_Registry.Insert
     (Integer (209), ErrorNoKernelImageForDevice'Identity);
   Exception_Registry.Insert (Integer (210), ErrorAlreadyAcquired'Identity);
   Exception_Registry.Insert (Integer (211), ErrorNotMapped'Identity);
   Exception_Registry.Insert (Integer (212), ErrorNotMappedAsArray'Identity);
   Exception_Registry.Insert (Integer (213), ErrorNotMappedAsPointer'Identity);
   Exception_Registry.Insert (Integer (214), ErrorECCUncorrectable'Identity);
   Exception_Registry.Insert (Integer (215), ErrorUnsupportedLimit'Identity);
   Exception_Registry.Insert (Integer (216), ErrorDeviceAlreadyInUse'Identity);
   Exception_Registry.Insert
     (Integer (217), ErrorPeerAccessUnsupported'Identity);
   Exception_Registry.Insert (Integer (218), ErrorInvalidPtx'Identity);
   Exception_Registry.Insert
     (Integer (219), ErrorInvalidGraphicsContext'Identity);
   Exception_Registry.Insert (Integer (220), ErrorNvlinkUncorrectable'Identity);
   Exception_Registry.Insert (Integer (221), ErrorJitCompilerNotFound'Identity);
   Exception_Registry.Insert (Integer (300), ErrorInvalidSource'Identity);
   Exception_Registry.Insert (Integer (301), ErrorFileNotFound'Identity);
   Exception_Registry.Insert
     (Integer (302), ErrorSharedObjectSymbolNotFound'Identity);
   Exception_Registry.Insert
     (Integer (303), ErrorSharedObjectInitFailed'Identity);
   Exception_Registry.Insert (Integer (304), ErrorOperatingSystem'Identity);
   Exception_Registry.Insert
     (Integer (400), ErrorInvalidResourceHandle'Identity);
   Exception_Registry.Insert (Integer (401), ErrorIllegalState'Identity);
   Exception_Registry.Insert (Integer (500), ErrorSymbolNotFound'Identity);
   Exception_Registry.Insert (Integer (600), ErrorNotReady'Identity);
   Exception_Registry.Insert (Integer (700), ErrorIllegalAddress'Identity);
   Exception_Registry.Insert
     (Integer (701), ErrorLaunchOutOfResources'Identity);
   Exception_Registry.Insert (Integer (702), ErrorLaunchTimeout'Identity);
   Exception_Registry.Insert
     (Integer (703), ErrorLaunchIncompatibleTexturing'Identity);
   Exception_Registry.Insert
     (Integer (704), ErrorPeerAccessAlreadyEnabled'Identity);
   Exception_Registry.Insert
     (Integer (705), ErrorPeerAccessNotEnabled'Identity);
   Exception_Registry.Insert (Integer (708), ErrorSetOnActiveProcess'Identity);
   Exception_Registry.Insert (Integer (709), ErrorContextIsDestroyed'Identity);
   Exception_Registry.Insert (Integer (710), ErrorAssert'Identity);
   Exception_Registry.Insert (Integer (711), ErrorTooManyPeers'Identity);
   Exception_Registry.Insert
     (Integer (712), ErrorHostMemoryAlreadyRegistered'Identity);
   Exception_Registry.Insert
     (Integer (713), ErrorHostMemoryNotRegistered'Identity);
   Exception_Registry.Insert (Integer (714), ErrorHardwareStackError'Identity);
   Exception_Registry.Insert (Integer (715), ErrorIllegalInstruction'Identity);
   Exception_Registry.Insert (Integer (716), ErrorMisalignedAddress'Identity);
   Exception_Registry.Insert (Integer (717), ErrorInvalidAddressSpace'Identity);
   Exception_Registry.Insert (Integer (718), ErrorInvalidPc'Identity);
   Exception_Registry.Insert (Integer (719), ErrorLaunchFailure'Identity);
   Exception_Registry.Insert
     (Integer (720), ErrorCooperativeLaunchTooLarge'Identity);
   Exception_Registry.Insert (Integer (800), ErrorNotPermitted'Identity);
   Exception_Registry.Insert (Integer (801), ErrorNotSupported'Identity);
   Exception_Registry.Insert (Integer (802), ErrorSystemNotReady'Identity);
   Exception_Registry.Insert
     (Integer (803), ErrorSystemDriverMismatch'Identity);
   Exception_Registry.Insert
     (Integer (804), ErrorCompatNotSupportedOnDevice'Identity);
   Exception_Registry.Insert
     (Integer (900), ErrorStreamCaptureUnsupported'Identity);
   Exception_Registry.Insert
     (Integer (901), ErrorStreamCaptureInvalidated'Identity);
   Exception_Registry.Insert (Integer (902), ErrorStreamCaptureMerge'Identity);
   Exception_Registry.Insert
     (Integer (903), ErrorStreamCaptureUnmatched'Identity);
   Exception_Registry.Insert
     (Integer (904), ErrorStreamCaptureUnjoined'Identity);
   Exception_Registry.Insert
     (Integer (905), ErrorStreamCaptureIsolation'Identity);
   Exception_Registry.Insert
     (Integer (906), ErrorStreamCaptureImplicit'Identity);
   Exception_Registry.Insert (Integer (907), ErrorCapturedEvent'Identity);
   Exception_Registry.Insert
     (Integer (908), ErrorStreamCaptureWrongThread'Identity);
   Exception_Registry.Insert (Integer (909), ErrorTimeout'Identity);
   Exception_Registry.Insert
     (Integer (910), ErrorGraphExecUpdateFailure'Identity);
   Exception_Registry.Insert (Integer (999), ErrorUnknown'Identity);
   Exception_Registry.Insert (Integer (10_000), ErrorApiFailureBase'Identity);

end CUDA;
