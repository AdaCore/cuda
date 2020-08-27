pragma Ada_2012;
pragma Style_Checks (Off);
pragma Warnings ("U");

with Interfaces.C; use Interfaces.C;
with udriver_types_h;
with stddef_h;
with Interfaces.C.Strings;
with System;
with Interfaces.C.Extensions;
with uvector_types_h;
with utexture_types_h;
with usurface_types_h;

package ucuda_runtime_api_h is

   CUDART_VERSION : constant := 11000;  --  /usr/local/cuda/include//cuda_runtime_api.h:136
   --  unsupported macro: CUDART_DEVICE __device__

   function cudaDeviceReset return udriver_types_h.cudaError_t  -- /usr/local/cuda/include//cuda_runtime_api.h:283
   with Import => True, 
        Convention => C, 
        External_Name => "cudaDeviceReset";

   function cudaDeviceSynchronize return udriver_types_h.cudaError_t  -- /usr/local/cuda/include//cuda_runtime_api.h:304
   with Import => True, 
        Convention => C, 
        External_Name => "cudaDeviceSynchronize";

   function cudaDeviceSetLimit (limit : udriver_types_h.cudaLimit; value : stddef_h.size_t) return udriver_types_h.cudaError_t  -- /usr/local/cuda/include//cuda_runtime_api.h:391
   with Import => True, 
        Convention => C, 
        External_Name => "cudaDeviceSetLimit";

   function cudaDeviceGetLimit (pValue : access stddef_h.size_t; limit : udriver_types_h.cudaLimit) return udriver_types_h.cudaError_t  -- /usr/local/cuda/include//cuda_runtime_api.h:426
   with Import => True, 
        Convention => C, 
        External_Name => "cudaDeviceGetLimit";

   function cudaDeviceGetCacheConfig (pCacheConfig : access udriver_types_h.cudaFuncCache) return udriver_types_h.cudaError_t  -- /usr/local/cuda/include//cuda_runtime_api.h:459
   with Import => True, 
        Convention => C, 
        External_Name => "cudaDeviceGetCacheConfig";

   function cudaDeviceGetStreamPriorityRange (leastPriority : access int; greatestPriority : access int) return udriver_types_h.cudaError_t  -- /usr/local/cuda/include//cuda_runtime_api.h:496
   with Import => True, 
        Convention => C, 
        External_Name => "cudaDeviceGetStreamPriorityRange";

   function cudaDeviceSetCacheConfig (cacheConfig : udriver_types_h.cudaFuncCache) return udriver_types_h.cudaError_t  -- /usr/local/cuda/include//cuda_runtime_api.h:540
   with Import => True, 
        Convention => C, 
        External_Name => "cudaDeviceSetCacheConfig";

   function cudaDeviceGetSharedMemConfig (pConfig : access udriver_types_h.cudaSharedMemConfig) return udriver_types_h.cudaError_t  -- /usr/local/cuda/include//cuda_runtime_api.h:571
   with Import => True, 
        Convention => C, 
        External_Name => "cudaDeviceGetSharedMemConfig";

   function cudaDeviceSetSharedMemConfig (config : udriver_types_h.cudaSharedMemConfig) return udriver_types_h.cudaError_t  -- /usr/local/cuda/include//cuda_runtime_api.h:615
   with Import => True, 
        Convention => C, 
        External_Name => "cudaDeviceSetSharedMemConfig";

   function cudaDeviceGetByPCIBusId (device : access int; pciBusId : Interfaces.C.Strings.chars_ptr) return udriver_types_h.cudaError_t  -- /usr/local/cuda/include//cuda_runtime_api.h:642
   with Import => True, 
        Convention => C, 
        External_Name => "cudaDeviceGetByPCIBusId";

   function cudaDeviceGetPCIBusId
     (pciBusId : Interfaces.C.Strings.chars_ptr;
      len : int;
      device : int) return udriver_types_h.cudaError_t  -- /usr/local/cuda/include//cuda_runtime_api.h:672
   with Import => True, 
        Convention => C, 
        External_Name => "cudaDeviceGetPCIBusId";

   function cudaIpcGetEventHandle (handle : access udriver_types_h.cudaIpcEventHandle_t; event : udriver_types_h.cudaEvent_t) return udriver_types_h.cudaError_t  -- /usr/local/cuda/include//cuda_runtime_api.h:720
   with Import => True, 
        Convention => C, 
        External_Name => "cudaIpcGetEventHandle";

   function cudaIpcOpenEventHandle (event : System.Address; handle : udriver_types_h.cudaIpcEventHandle_t) return udriver_types_h.cudaError_t  -- /usr/local/cuda/include//cuda_runtime_api.h:761
   with Import => True, 
        Convention => C, 
        External_Name => "cudaIpcOpenEventHandle";

   function cudaIpcGetMemHandle (handle : access udriver_types_h.cudaIpcMemHandle_t; devPtr : System.Address) return udriver_types_h.cudaError_t  -- /usr/local/cuda/include//cuda_runtime_api.h:804
   with Import => True, 
        Convention => C, 
        External_Name => "cudaIpcGetMemHandle";

   function cudaIpcOpenMemHandle
     (devPtr : System.Address;
      handle : udriver_types_h.cudaIpcMemHandle_t;
      flags : unsigned) return udriver_types_h.cudaError_t  -- /usr/local/cuda/include//cuda_runtime_api.h:864
   with Import => True, 
        Convention => C, 
        External_Name => "cudaIpcOpenMemHandle";

   function cudaIpcCloseMemHandle (devPtr : System.Address) return udriver_types_h.cudaError_t  -- /usr/local/cuda/include//cuda_runtime_api.h:899
   with Import => True, 
        Convention => C, 
        External_Name => "cudaIpcCloseMemHandle";

   function cudaThreadExit return udriver_types_h.cudaError_t  -- /usr/local/cuda/include//cuda_runtime_api.h:941
   with Import => True, 
        Convention => C, 
        External_Name => "cudaThreadExit";

   function cudaThreadSynchronize return udriver_types_h.cudaError_t  -- /usr/local/cuda/include//cuda_runtime_api.h:967
   with Import => True, 
        Convention => C, 
        External_Name => "cudaThreadSynchronize";

   function cudaThreadSetLimit (limit : udriver_types_h.cudaLimit; value : stddef_h.size_t) return udriver_types_h.cudaError_t  -- /usr/local/cuda/include//cuda_runtime_api.h:1016
   with Import => True, 
        Convention => C, 
        External_Name => "cudaThreadSetLimit";

   function cudaThreadGetLimit (pValue : access stddef_h.size_t; limit : udriver_types_h.cudaLimit) return udriver_types_h.cudaError_t  -- /usr/local/cuda/include//cuda_runtime_api.h:1049
   with Import => True, 
        Convention => C, 
        External_Name => "cudaThreadGetLimit";

   function cudaThreadGetCacheConfig (pCacheConfig : access udriver_types_h.cudaFuncCache) return udriver_types_h.cudaError_t  -- /usr/local/cuda/include//cuda_runtime_api.h:1085
   with Import => True, 
        Convention => C, 
        External_Name => "cudaThreadGetCacheConfig";

   function cudaThreadSetCacheConfig (cacheConfig : udriver_types_h.cudaFuncCache) return udriver_types_h.cudaError_t  -- /usr/local/cuda/include//cuda_runtime_api.h:1132
   with Import => True, 
        Convention => C, 
        External_Name => "cudaThreadSetCacheConfig";

   function cudaGetLastError return udriver_types_h.cudaError_t  -- /usr/local/cuda/include//cuda_runtime_api.h:1191
   with Import => True, 
        Convention => C, 
        External_Name => "cudaGetLastError";

   function cudaPeekAtLastError return udriver_types_h.cudaError_t  -- /usr/local/cuda/include//cuda_runtime_api.h:1237
   with Import => True, 
        Convention => C, 
        External_Name => "cudaPeekAtLastError";

   function cudaGetErrorName (arg1 : udriver_types_h.cudaError_t) return Interfaces.C.Strings.chars_ptr  -- /usr/local/cuda/include//cuda_runtime_api.h:1253
   with Import => True, 
        Convention => C, 
        External_Name => "cudaGetErrorName";

   function cudaGetErrorString (arg1 : udriver_types_h.cudaError_t) return Interfaces.C.Strings.chars_ptr  -- /usr/local/cuda/include//cuda_runtime_api.h:1269
   with Import => True, 
        Convention => C, 
        External_Name => "cudaGetErrorString";

   function cudaGetDeviceCount (count : access int) return udriver_types_h.cudaError_t  -- /usr/local/cuda/include//cuda_runtime_api.h:1297
   with Import => True, 
        Convention => C, 
        External_Name => "cudaGetDeviceCount";

   function cudaGetDeviceProperties (prop : access udriver_types_h.cudaDeviceProp; device : int) return udriver_types_h.cudaError_t  -- /usr/local/cuda/include//cuda_runtime_api.h:1575
   with Import => True, 
        Convention => C, 
        External_Name => "cudaGetDeviceProperties";

   function cudaDeviceGetAttribute
     (value : access int;
      attr : udriver_types_h.cudaDeviceAttr;
      device : int) return udriver_types_h.cudaError_t  -- /usr/local/cuda/include//cuda_runtime_api.h:1765
   with Import => True, 
        Convention => C, 
        External_Name => "cudaDeviceGetAttribute";

   function cudaDeviceGetNvSciSyncAttributes
     (nvSciSyncAttrList : System.Address;
      device : int;
      flags : int) return udriver_types_h.cudaError_t  -- /usr/local/cuda/include//cuda_runtime_api.h:1814
   with Import => True, 
        Convention => C, 
        External_Name => "cudaDeviceGetNvSciSyncAttributes";

   function cudaDeviceGetP2PAttribute
     (value : access int;
      attr : udriver_types_h.cudaDeviceP2PAttr;
      srcDevice : int;
      dstDevice : int) return udriver_types_h.cudaError_t  -- /usr/local/cuda/include//cuda_runtime_api.h:1854
   with Import => True, 
        Convention => C, 
        External_Name => "cudaDeviceGetP2PAttribute";

   function cudaChooseDevice (device : access int; prop : access constant udriver_types_h.cudaDeviceProp) return udriver_types_h.cudaError_t  -- /usr/local/cuda/include//cuda_runtime_api.h:1875
   with Import => True, 
        Convention => C, 
        External_Name => "cudaChooseDevice";

   function cudaSetDevice (device : int) return udriver_types_h.cudaError_t  -- /usr/local/cuda/include//cuda_runtime_api.h:1912
   with Import => True, 
        Convention => C, 
        External_Name => "cudaSetDevice";

   function cudaGetDevice (device : access int) return udriver_types_h.cudaError_t  -- /usr/local/cuda/include//cuda_runtime_api.h:1933
   with Import => True, 
        Convention => C, 
        External_Name => "cudaGetDevice";

   function cudaSetValidDevices (device_arr : access int; len : int) return udriver_types_h.cudaError_t  -- /usr/local/cuda/include//cuda_runtime_api.h:1964
   with Import => True, 
        Convention => C, 
        External_Name => "cudaSetValidDevices";

   function cudaSetDeviceFlags (flags : unsigned) return udriver_types_h.cudaError_t  -- /usr/local/cuda/include//cuda_runtime_api.h:2033
   with Import => True, 
        Convention => C, 
        External_Name => "cudaSetDeviceFlags";

   function cudaGetDeviceFlags (flags : access unsigned) return udriver_types_h.cudaError_t  -- /usr/local/cuda/include//cuda_runtime_api.h:2079
   with Import => True, 
        Convention => C, 
        External_Name => "cudaGetDeviceFlags";

   function cudaStreamCreate (pStream : System.Address) return udriver_types_h.cudaError_t  -- /usr/local/cuda/include//cuda_runtime_api.h:2119
   with Import => True, 
        Convention => C, 
        External_Name => "cudaStreamCreate";

   function cudaStreamCreateWithFlags (pStream : System.Address; flags : unsigned) return udriver_types_h.cudaError_t  -- /usr/local/cuda/include//cuda_runtime_api.h:2151
   with Import => True, 
        Convention => C, 
        External_Name => "cudaStreamCreateWithFlags";

   function cudaStreamCreateWithPriority
     (pStream : System.Address;
      flags : unsigned;
      priority : int) return udriver_types_h.cudaError_t  -- /usr/local/cuda/include//cuda_runtime_api.h:2197
   with Import => True, 
        Convention => C, 
        External_Name => "cudaStreamCreateWithPriority";

   function cudaStreamGetPriority (hStream : udriver_types_h.cudaStream_t; priority : access int) return udriver_types_h.cudaError_t  -- /usr/local/cuda/include//cuda_runtime_api.h:2224
   with Import => True, 
        Convention => C, 
        External_Name => "cudaStreamGetPriority";

   function cudaStreamGetFlags (hStream : udriver_types_h.cudaStream_t; flags : access unsigned) return udriver_types_h.cudaError_t  -- /usr/local/cuda/include//cuda_runtime_api.h:2249
   with Import => True, 
        Convention => C, 
        External_Name => "cudaStreamGetFlags";

   function cudaCtxResetPersistingL2Cache return udriver_types_h.cudaError_t  -- /usr/local/cuda/include//cuda_runtime_api.h:2264
   with Import => True, 
        Convention => C, 
        External_Name => "cudaCtxResetPersistingL2Cache";

   function cudaStreamCopyAttributes (dst : udriver_types_h.cudaStream_t; src : udriver_types_h.cudaStream_t) return udriver_types_h.cudaError_t  -- /usr/local/cuda/include//cuda_runtime_api.h:2284
   with Import => True, 
        Convention => C, 
        External_Name => "cudaStreamCopyAttributes";

   function cudaStreamGetAttribute
     (hStream : udriver_types_h.cudaStream_t;
      attr : udriver_types_h.cudaStreamAttrID;
      value_out : access udriver_types_h.cudaStreamAttrValue) return udriver_types_h.cudaError_t  -- /usr/local/cuda/include//cuda_runtime_api.h:2305
   with Import => True, 
        Convention => C, 
        External_Name => "cudaStreamGetAttribute";

   function cudaStreamSetAttribute
     (hStream : udriver_types_h.cudaStream_t;
      attr : udriver_types_h.cudaStreamAttrID;
      value : access constant udriver_types_h.cudaStreamAttrValue) return udriver_types_h.cudaError_t  -- /usr/local/cuda/include//cuda_runtime_api.h:2329
   with Import => True, 
        Convention => C, 
        External_Name => "cudaStreamSetAttribute";

   function cudaStreamDestroy (stream : udriver_types_h.cudaStream_t) return udriver_types_h.cudaError_t  -- /usr/local/cuda/include//cuda_runtime_api.h:2362
   with Import => True, 
        Convention => C, 
        External_Name => "cudaStreamDestroy";

   function cudaStreamWaitEvent
     (stream : udriver_types_h.cudaStream_t;
      event : udriver_types_h.cudaEvent_t;
      flags : unsigned) return udriver_types_h.cudaError_t  -- /usr/local/cuda/include//cuda_runtime_api.h:2388
   with Import => True, 
        Convention => C, 
        External_Name => "cudaStreamWaitEvent";

   type cudaStreamCallback_t is access procedure
        (arg1 : udriver_types_h.cudaStream_t;
         arg2 : udriver_types_h.cudaError_t;
         arg3 : System.Address)
   with Convention => C;  -- /usr/local/cuda/include//cuda_runtime_api.h:2396

   function cudaStreamAddCallback
     (stream : udriver_types_h.cudaStream_t;
      callback : cudaStreamCallback_t;
      userData : System.Address;
      flags : unsigned) return udriver_types_h.cudaError_t  -- /usr/local/cuda/include//cuda_runtime_api.h:2463
   with Import => True, 
        Convention => C, 
        External_Name => "cudaStreamAddCallback";

   function cudaStreamSynchronize (stream : udriver_types_h.cudaStream_t) return udriver_types_h.cudaError_t  -- /usr/local/cuda/include//cuda_runtime_api.h:2487
   with Import => True, 
        Convention => C, 
        External_Name => "cudaStreamSynchronize";

   function cudaStreamQuery (stream : udriver_types_h.cudaStream_t) return udriver_types_h.cudaError_t  -- /usr/local/cuda/include//cuda_runtime_api.h:2512
   with Import => True, 
        Convention => C, 
        External_Name => "cudaStreamQuery";

   function cudaStreamAttachMemAsync
     (stream : udriver_types_h.cudaStream_t;
      devPtr : System.Address;
      length : stddef_h.size_t;
      flags : unsigned) return udriver_types_h.cudaError_t  -- /usr/local/cuda/include//cuda_runtime_api.h:2595
   with Import => True, 
        Convention => C, 
        External_Name => "cudaStreamAttachMemAsync";

   function cudaStreamBeginCapture (stream : udriver_types_h.cudaStream_t; mode : udriver_types_h.cudaStreamCaptureMode) return udriver_types_h.cudaError_t  -- /usr/local/cuda/include//cuda_runtime_api.h:2631
   with Import => True, 
        Convention => C, 
        External_Name => "cudaStreamBeginCapture";

   function cudaThreadExchangeStreamCaptureMode (mode : access udriver_types_h.cudaStreamCaptureMode) return udriver_types_h.cudaError_t  -- /usr/local/cuda/include//cuda_runtime_api.h:2682
   with Import => True, 
        Convention => C, 
        External_Name => "cudaThreadExchangeStreamCaptureMode";

   function cudaStreamEndCapture (stream : udriver_types_h.cudaStream_t; pGraph : System.Address) return udriver_types_h.cudaError_t  -- /usr/local/cuda/include//cuda_runtime_api.h:2710
   with Import => True, 
        Convention => C, 
        External_Name => "cudaStreamEndCapture";

   function cudaStreamIsCapturing (stream : udriver_types_h.cudaStream_t; pCaptureStatus : access udriver_types_h.cudaStreamCaptureStatus) return udriver_types_h.cudaError_t  -- /usr/local/cuda/include//cuda_runtime_api.h:2748
   with Import => True, 
        Convention => C, 
        External_Name => "cudaStreamIsCapturing";

   function cudaStreamGetCaptureInfo
     (stream : udriver_types_h.cudaStream_t;
      pCaptureStatus : access udriver_types_h.cudaStreamCaptureStatus;
      pId : access Extensions.unsigned_long_long) return udriver_types_h.cudaError_t  -- /usr/local/cuda/include//cuda_runtime_api.h:2776
   with Import => True, 
        Convention => C, 
        External_Name => "cudaStreamGetCaptureInfo";

   function cudaEventCreate (event : System.Address) return udriver_types_h.cudaError_t  -- /usr/local/cuda/include//cuda_runtime_api.h:2813
   with Import => True, 
        Convention => C, 
        External_Name => "cudaEventCreate";

   function cudaEventCreateWithFlags (event : System.Address; flags : unsigned) return udriver_types_h.cudaError_t  -- /usr/local/cuda/include//cuda_runtime_api.h:2850
   with Import => True, 
        Convention => C, 
        External_Name => "cudaEventCreateWithFlags";

   function cudaEventRecord (event : udriver_types_h.cudaEvent_t; stream : udriver_types_h.cudaStream_t) return udriver_types_h.cudaError_t  -- /usr/local/cuda/include//cuda_runtime_api.h:2889
   with Import => True, 
        Convention => C, 
        External_Name => "cudaEventRecord";

   function cudaEventQuery (event : udriver_types_h.cudaEvent_t) return udriver_types_h.cudaError_t  -- /usr/local/cuda/include//cuda_runtime_api.h:2920
   with Import => True, 
        Convention => C, 
        External_Name => "cudaEventQuery";

   function cudaEventSynchronize (event : udriver_types_h.cudaEvent_t) return udriver_types_h.cudaError_t  -- /usr/local/cuda/include//cuda_runtime_api.h:2950
   with Import => True, 
        Convention => C, 
        External_Name => "cudaEventSynchronize";

   function cudaEventDestroy (event : udriver_types_h.cudaEvent_t) return udriver_types_h.cudaError_t  -- /usr/local/cuda/include//cuda_runtime_api.h:2978
   with Import => True, 
        Convention => C, 
        External_Name => "cudaEventDestroy";

   function cudaEventElapsedTime
     (ms : access float;
      start : udriver_types_h.cudaEvent_t;
      c_end : udriver_types_h.cudaEvent_t) return udriver_types_h.cudaError_t  -- /usr/local/cuda/include//cuda_runtime_api.h:3021
   with Import => True, 
        Convention => C, 
        External_Name => "cudaEventElapsedTime";

   function cudaImportExternalMemory (extMem_out : System.Address; memHandleDesc : access constant udriver_types_h.cudaExternalMemoryHandleDesc) return udriver_types_h.cudaError_t  -- /usr/local/cuda/include//cuda_runtime_api.h:3198
   with Import => True, 
        Convention => C, 
        External_Name => "cudaImportExternalMemory";

   function cudaExternalMemoryGetMappedBuffer
     (devPtr : System.Address;
      extMem : udriver_types_h.cudaExternalMemory_t;
      bufferDesc : access constant udriver_types_h.cudaExternalMemoryBufferDesc) return udriver_types_h.cudaError_t  -- /usr/local/cuda/include//cuda_runtime_api.h:3252
   with Import => True, 
        Convention => C, 
        External_Name => "cudaExternalMemoryGetMappedBuffer";

   function cudaExternalMemoryGetMappedMipmappedArray
     (mipmap : System.Address;
      extMem : udriver_types_h.cudaExternalMemory_t;
      mipmapDesc : access constant udriver_types_h.cudaExternalMemoryMipmappedArrayDesc) return udriver_types_h.cudaError_t  -- /usr/local/cuda/include//cuda_runtime_api.h:3311
   with Import => True, 
        Convention => C, 
        External_Name => "cudaExternalMemoryGetMappedMipmappedArray";

   function cudaDestroyExternalMemory (extMem : udriver_types_h.cudaExternalMemory_t) return udriver_types_h.cudaError_t  -- /usr/local/cuda/include//cuda_runtime_api.h:3334
   with Import => True, 
        Convention => C, 
        External_Name => "cudaDestroyExternalMemory";

   function cudaImportExternalSemaphore (extSem_out : System.Address; semHandleDesc : access constant udriver_types_h.cudaExternalSemaphoreHandleDesc) return udriver_types_h.cudaError_t  -- /usr/local/cuda/include//cuda_runtime_api.h:3465
   with Import => True, 
        Convention => C, 
        External_Name => "cudaImportExternalSemaphore";

   function cudaSignalExternalSemaphoresAsync
     (extSemArray : System.Address;
      paramsArray : access constant udriver_types_h.cudaExternalSemaphoreSignalParams;
      numExtSems : unsigned;
      stream : udriver_types_h.cudaStream_t) return udriver_types_h.cudaError_t  -- /usr/local/cuda/include//cuda_runtime_api.h:3530
   with Import => True, 
        Convention => C, 
        External_Name => "cudaSignalExternalSemaphoresAsync";

   function cudaWaitExternalSemaphoresAsync
     (extSemArray : System.Address;
      paramsArray : access constant udriver_types_h.cudaExternalSemaphoreWaitParams;
      numExtSems : unsigned;
      stream : udriver_types_h.cudaStream_t) return udriver_types_h.cudaError_t  -- /usr/local/cuda/include//cuda_runtime_api.h:3604
   with Import => True, 
        Convention => C, 
        External_Name => "cudaWaitExternalSemaphoresAsync";

   function cudaDestroyExternalSemaphore (extSem : udriver_types_h.cudaExternalSemaphore_t) return udriver_types_h.cudaError_t  -- /usr/local/cuda/include//cuda_runtime_api.h:3626
   with Import => True, 
        Convention => C, 
        External_Name => "cudaDestroyExternalSemaphore";

   function cudaLaunchKernel
     (func : System.Address;
      gridDim : uvector_types_h.dim3;
      blockDim : uvector_types_h.dim3;
      args : System.Address;
      sharedMem : stddef_h.size_t;
      stream : udriver_types_h.cudaStream_t) return udriver_types_h.cudaError_t  -- /usr/local/cuda/include//cuda_runtime_api.h:3691
   with Import => True, 
        Convention => C, 
        External_Name => "cudaLaunchKernel";

   function cudaLaunchCooperativeKernel
     (func : System.Address;
      gridDim : uvector_types_h.dim3;
      blockDim : uvector_types_h.dim3;
      args : System.Address;
      sharedMem : stddef_h.size_t;
      stream : udriver_types_h.cudaStream_t) return udriver_types_h.cudaError_t  -- /usr/local/cuda/include//cuda_runtime_api.h:3748
   with Import => True, 
        Convention => C, 
        External_Name => "cudaLaunchCooperativeKernel";

   function cudaLaunchCooperativeKernelMultiDevice
     (launchParamsList : access udriver_types_h.cudaLaunchParams;
      numDevices : unsigned;
      flags : unsigned) return udriver_types_h.cudaError_t  -- /usr/local/cuda/include//cuda_runtime_api.h:3847
   with Import => True, 
        Convention => C, 
        External_Name => "cudaLaunchCooperativeKernelMultiDevice";

   function cudaFuncSetCacheConfig (func : System.Address; cacheConfig : udriver_types_h.cudaFuncCache) return udriver_types_h.cudaError_t  -- /usr/local/cuda/include//cuda_runtime_api.h:3896
   with Import => True, 
        Convention => C, 
        External_Name => "cudaFuncSetCacheConfig";

   function cudaFuncSetSharedMemConfig (func : System.Address; config : udriver_types_h.cudaSharedMemConfig) return udriver_types_h.cudaError_t  -- /usr/local/cuda/include//cuda_runtime_api.h:3951
   with Import => True, 
        Convention => C, 
        External_Name => "cudaFuncSetSharedMemConfig";

   function cudaFuncGetAttributes (attr : access udriver_types_h.cudaFuncAttributes; func : System.Address) return udriver_types_h.cudaError_t  -- /usr/local/cuda/include//cuda_runtime_api.h:3986
   with Import => True, 
        Convention => C, 
        External_Name => "cudaFuncGetAttributes";

   function cudaFuncSetAttribute
     (func : System.Address;
      attr : udriver_types_h.cudaFuncAttribute;
      value : int) return udriver_types_h.cudaError_t  -- /usr/local/cuda/include//cuda_runtime_api.h:4025
   with Import => True, 
        Convention => C, 
        External_Name => "cudaFuncSetAttribute";

   function cudaSetDoubleForDevice (d : access double) return udriver_types_h.cudaError_t  -- /usr/local/cuda/include//cuda_runtime_api.h:4049
   with Import => True, 
        Convention => C, 
        External_Name => "cudaSetDoubleForDevice";

   function cudaSetDoubleForHost (d : access double) return udriver_types_h.cudaError_t  -- /usr/local/cuda/include//cuda_runtime_api.h:4073
   with Import => True, 
        Convention => C, 
        External_Name => "cudaSetDoubleForHost";

   function cudaLaunchHostFunc
     (stream : udriver_types_h.cudaStream_t;
      fn : udriver_types_h.cudaHostFn_t;
      userData : System.Address) return udriver_types_h.cudaError_t  -- /usr/local/cuda/include//cuda_runtime_api.h:4139
   with Import => True, 
        Convention => C, 
        External_Name => "cudaLaunchHostFunc";

   function cudaOccupancyMaxActiveBlocksPerMultiprocessor
     (numBlocks : access int;
      func : System.Address;
      blockSize : int;
      dynamicSMemSize : stddef_h.size_t) return udriver_types_h.cudaError_t  -- /usr/local/cuda/include//cuda_runtime_api.h:4196
   with Import => True, 
        Convention => C, 
        External_Name => "cudaOccupancyMaxActiveBlocksPerMultiprocessor";

   function cudaOccupancyAvailableDynamicSMemPerBlock
     (dynamicSmemSize : access stddef_h.size_t;
      func : System.Address;
      numBlocks : int;
      blockSize : int) return udriver_types_h.cudaError_t  -- /usr/local/cuda/include//cuda_runtime_api.h:4225
   with Import => True, 
        Convention => C, 
        External_Name => "cudaOccupancyAvailableDynamicSMemPerBlock";

   function cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags
     (numBlocks : access int;
      func : System.Address;
      blockSize : int;
      dynamicSMemSize : stddef_h.size_t;
      flags : unsigned) return udriver_types_h.cudaError_t  -- /usr/local/cuda/include//cuda_runtime_api.h:4270
   with Import => True, 
        Convention => C, 
        External_Name => "cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags";

   function cudaMallocManaged
     (devPtr : System.Address;
      size : stddef_h.size_t;
      flags : unsigned) return udriver_types_h.cudaError_t  -- /usr/local/cuda/include//cuda_runtime_api.h:4390
   with Import => True, 
        Convention => C, 
        External_Name => "cudaMallocManaged";

   function cudaMalloc (devPtr : System.Address; size : stddef_h.size_t) return udriver_types_h.cudaError_t  -- /usr/local/cuda/include//cuda_runtime_api.h:4421
   with Import => True, 
        Convention => C, 
        External_Name => "cudaMalloc";

   function cudaMallocHost (ptr : System.Address; size : stddef_h.size_t) return udriver_types_h.cudaError_t  -- /usr/local/cuda/include//cuda_runtime_api.h:4454
   with Import => True, 
        Convention => C, 
        External_Name => "cudaMallocHost";

   function cudaMallocPitch
     (devPtr : System.Address;
      pitch : access stddef_h.size_t;
      width : stddef_h.size_t;
      height : stddef_h.size_t) return udriver_types_h.cudaError_t  -- /usr/local/cuda/include//cuda_runtime_api.h:4497
   with Import => True, 
        Convention => C, 
        External_Name => "cudaMallocPitch";

   function cudaMallocArray
     (c_array : System.Address;
      desc : access constant udriver_types_h.cudaChannelFormatDesc;
      width : stddef_h.size_t;
      height : stddef_h.size_t;
      flags : unsigned) return udriver_types_h.cudaError_t  -- /usr/local/cuda/include//cuda_runtime_api.h:4543
   with Import => True, 
        Convention => C, 
        External_Name => "cudaMallocArray";

   function cudaFree (devPtr : System.Address) return udriver_types_h.cudaError_t  -- /usr/local/cuda/include//cuda_runtime_api.h:4572
   with Import => True, 
        Convention => C, 
        External_Name => "cudaFree";

   function cudaFreeHost (ptr : System.Address) return udriver_types_h.cudaError_t  -- /usr/local/cuda/include//cuda_runtime_api.h:4595
   with Import => True, 
        Convention => C, 
        External_Name => "cudaFreeHost";

   function cudaFreeArray (c_array : udriver_types_h.cudaArray_t) return udriver_types_h.cudaError_t  -- /usr/local/cuda/include//cuda_runtime_api.h:4618
   with Import => True, 
        Convention => C, 
        External_Name => "cudaFreeArray";

   function cudaFreeMipmappedArray (mipmappedArray : udriver_types_h.cudaMipmappedArray_t) return udriver_types_h.cudaError_t  -- /usr/local/cuda/include//cuda_runtime_api.h:4641
   with Import => True, 
        Convention => C, 
        External_Name => "cudaFreeMipmappedArray";

   function cudaHostAlloc
     (pHost : System.Address;
      size : stddef_h.size_t;
      flags : unsigned) return udriver_types_h.cudaError_t  -- /usr/local/cuda/include//cuda_runtime_api.h:4707
   with Import => True, 
        Convention => C, 
        External_Name => "cudaHostAlloc";

   function cudaHostRegister
     (ptr : System.Address;
      size : stddef_h.size_t;
      flags : unsigned) return udriver_types_h.cudaError_t  -- /usr/local/cuda/include//cuda_runtime_api.h:4791
   with Import => True, 
        Convention => C, 
        External_Name => "cudaHostRegister";

   function cudaHostUnregister (ptr : System.Address) return udriver_types_h.cudaError_t  -- /usr/local/cuda/include//cuda_runtime_api.h:4814
   with Import => True, 
        Convention => C, 
        External_Name => "cudaHostUnregister";

   function cudaHostGetDevicePointer
     (pDevice : System.Address;
      pHost : System.Address;
      flags : unsigned) return udriver_types_h.cudaError_t  -- /usr/local/cuda/include//cuda_runtime_api.h:4859
   with Import => True, 
        Convention => C, 
        External_Name => "cudaHostGetDevicePointer";

   function cudaHostGetFlags (pFlags : access unsigned; pHost : System.Address) return udriver_types_h.cudaError_t  -- /usr/local/cuda/include//cuda_runtime_api.h:4881
   with Import => True, 
        Convention => C, 
        External_Name => "cudaHostGetFlags";

   function cudaMalloc3D (pitchedDevPtr : access udriver_types_h.cudaPitchedPtr; extent : udriver_types_h.cudaExtent) return udriver_types_h.cudaError_t  -- /usr/local/cuda/include//cuda_runtime_api.h:4920
   with Import => True, 
        Convention => C, 
        External_Name => "cudaMalloc3D";

   function cudaMalloc3DArray
     (c_array : System.Address;
      desc : access constant udriver_types_h.cudaChannelFormatDesc;
      extent : udriver_types_h.cudaExtent;
      flags : unsigned) return udriver_types_h.cudaError_t  -- /usr/local/cuda/include//cuda_runtime_api.h:5059
   with Import => True, 
        Convention => C, 
        External_Name => "cudaMalloc3DArray";

   function cudaMallocMipmappedArray
     (mipmappedArray : System.Address;
      desc : access constant udriver_types_h.cudaChannelFormatDesc;
      extent : udriver_types_h.cudaExtent;
      numLevels : unsigned;
      flags : unsigned) return udriver_types_h.cudaError_t  -- /usr/local/cuda/include//cuda_runtime_api.h:5198
   with Import => True, 
        Convention => C, 
        External_Name => "cudaMallocMipmappedArray";

   function cudaGetMipmappedArrayLevel
     (levelArray : System.Address;
      mipmappedArray : udriver_types_h.cudaMipmappedArray_const_t;
      level : unsigned) return udriver_types_h.cudaError_t  -- /usr/local/cuda/include//cuda_runtime_api.h:5231
   with Import => True, 
        Convention => C, 
        External_Name => "cudaGetMipmappedArrayLevel";

   function cudaMemcpy3D (p : access constant udriver_types_h.cudaMemcpy3DParms) return udriver_types_h.cudaError_t  -- /usr/local/cuda/include//cuda_runtime_api.h:5336
   with Import => True, 
        Convention => C, 
        External_Name => "cudaMemcpy3D";

   function cudaMemcpy3DPeer (p : access constant udriver_types_h.cudaMemcpy3DPeerParms) return udriver_types_h.cudaError_t  -- /usr/local/cuda/include//cuda_runtime_api.h:5367
   with Import => True, 
        Convention => C, 
        External_Name => "cudaMemcpy3DPeer";

   function cudaMemcpy3DAsync (p : access constant udriver_types_h.cudaMemcpy3DParms; stream : udriver_types_h.cudaStream_t) return udriver_types_h.cudaError_t  -- /usr/local/cuda/include//cuda_runtime_api.h:5485
   with Import => True, 
        Convention => C, 
        External_Name => "cudaMemcpy3DAsync";

   function cudaMemcpy3DPeerAsync (p : access constant udriver_types_h.cudaMemcpy3DPeerParms; stream : udriver_types_h.cudaStream_t) return udriver_types_h.cudaError_t  -- /usr/local/cuda/include//cuda_runtime_api.h:5511
   with Import => True, 
        Convention => C, 
        External_Name => "cudaMemcpy3DPeerAsync";

   function cudaMemGetInfo (free : access stddef_h.size_t; total : access stddef_h.size_t) return udriver_types_h.cudaError_t  -- /usr/local/cuda/include//cuda_runtime_api.h:5533
   with Import => True, 
        Convention => C, 
        External_Name => "cudaMemGetInfo";

   function cudaArrayGetInfo
     (desc : access udriver_types_h.cudaChannelFormatDesc;
      extent : access udriver_types_h.cudaExtent;
      flags : access unsigned;
      c_array : udriver_types_h.cudaArray_t) return udriver_types_h.cudaError_t  -- /usr/local/cuda/include//cuda_runtime_api.h:5559
   with Import => True, 
        Convention => C, 
        External_Name => "cudaArrayGetInfo";

   function cudaMemcpy
     (dst : System.Address;
      src : System.Address;
      count : stddef_h.size_t;
      kind : udriver_types_h.cudaMemcpyKind) return udriver_types_h.cudaError_t  -- /usr/local/cuda/include//cuda_runtime_api.h:5603
   with Import => True, 
        Convention => C, 
        External_Name => "cudaMemcpy";

   function cudaMemcpyPeer
     (dst : System.Address;
      dstDevice : int;
      src : System.Address;
      srcDevice : int;
      count : stddef_h.size_t) return udriver_types_h.cudaError_t  -- /usr/local/cuda/include//cuda_runtime_api.h:5638
   with Import => True, 
        Convention => C, 
        External_Name => "cudaMemcpyPeer";

   function cudaMemcpy2D
     (dst : System.Address;
      dpitch : stddef_h.size_t;
      src : System.Address;
      spitch : stddef_h.size_t;
      width : stddef_h.size_t;
      height : stddef_h.size_t;
      kind : udriver_types_h.cudaMemcpyKind) return udriver_types_h.cudaError_t  -- /usr/local/cuda/include//cuda_runtime_api.h:5687
   with Import => True, 
        Convention => C, 
        External_Name => "cudaMemcpy2D";

   function cudaMemcpy2DToArray
     (dst : udriver_types_h.cudaArray_t;
      wOffset : stddef_h.size_t;
      hOffset : stddef_h.size_t;
      src : System.Address;
      spitch : stddef_h.size_t;
      width : stddef_h.size_t;
      height : stddef_h.size_t;
      kind : udriver_types_h.cudaMemcpyKind) return udriver_types_h.cudaError_t  -- /usr/local/cuda/include//cuda_runtime_api.h:5737
   with Import => True, 
        Convention => C, 
        External_Name => "cudaMemcpy2DToArray";

   function cudaMemcpy2DFromArray
     (dst : System.Address;
      dpitch : stddef_h.size_t;
      src : udriver_types_h.cudaArray_const_t;
      wOffset : stddef_h.size_t;
      hOffset : stddef_h.size_t;
      width : stddef_h.size_t;
      height : stddef_h.size_t;
      kind : udriver_types_h.cudaMemcpyKind) return udriver_types_h.cudaError_t  -- /usr/local/cuda/include//cuda_runtime_api.h:5787
   with Import => True, 
        Convention => C, 
        External_Name => "cudaMemcpy2DFromArray";

   function cudaMemcpy2DArrayToArray
     (dst : udriver_types_h.cudaArray_t;
      wOffsetDst : stddef_h.size_t;
      hOffsetDst : stddef_h.size_t;
      src : udriver_types_h.cudaArray_const_t;
      wOffsetSrc : stddef_h.size_t;
      hOffsetSrc : stddef_h.size_t;
      width : stddef_h.size_t;
      height : stddef_h.size_t;
      kind : udriver_types_h.cudaMemcpyKind) return udriver_types_h.cudaError_t  -- /usr/local/cuda/include//cuda_runtime_api.h:5834
   with Import => True, 
        Convention => C, 
        External_Name => "cudaMemcpy2DArrayToArray";

   function cudaMemcpyToSymbol
     (symbol : System.Address;
      src : System.Address;
      count : stddef_h.size_t;
      offset : stddef_h.size_t;
      kind : udriver_types_h.cudaMemcpyKind) return udriver_types_h.cudaError_t  -- /usr/local/cuda/include//cuda_runtime_api.h:5877
   with Import => True, 
        Convention => C, 
        External_Name => "cudaMemcpyToSymbol";

   function cudaMemcpyFromSymbol
     (dst : System.Address;
      symbol : System.Address;
      count : stddef_h.size_t;
      offset : stddef_h.size_t;
      kind : udriver_types_h.cudaMemcpyKind) return udriver_types_h.cudaError_t  -- /usr/local/cuda/include//cuda_runtime_api.h:5920
   with Import => True, 
        Convention => C, 
        External_Name => "cudaMemcpyFromSymbol";

   function cudaMemcpyAsync
     (dst : System.Address;
      src : System.Address;
      count : stddef_h.size_t;
      kind : udriver_types_h.cudaMemcpyKind;
      stream : udriver_types_h.cudaStream_t) return udriver_types_h.cudaError_t  -- /usr/local/cuda/include//cuda_runtime_api.h:5977
   with Import => True, 
        Convention => C, 
        External_Name => "cudaMemcpyAsync";

   function cudaMemcpyPeerAsync
     (dst : System.Address;
      dstDevice : int;
      src : System.Address;
      srcDevice : int;
      count : stddef_h.size_t;
      stream : udriver_types_h.cudaStream_t) return udriver_types_h.cudaError_t  -- /usr/local/cuda/include//cuda_runtime_api.h:6012
   with Import => True, 
        Convention => C, 
        External_Name => "cudaMemcpyPeerAsync";

   function cudaMemcpy2DAsync
     (dst : System.Address;
      dpitch : stddef_h.size_t;
      src : System.Address;
      spitch : stddef_h.size_t;
      width : stddef_h.size_t;
      height : stddef_h.size_t;
      kind : udriver_types_h.cudaMemcpyKind;
      stream : udriver_types_h.cudaStream_t) return udriver_types_h.cudaError_t  -- /usr/local/cuda/include//cuda_runtime_api.h:6075
   with Import => True, 
        Convention => C, 
        External_Name => "cudaMemcpy2DAsync";

   function cudaMemcpy2DToArrayAsync
     (dst : udriver_types_h.cudaArray_t;
      wOffset : stddef_h.size_t;
      hOffset : stddef_h.size_t;
      src : System.Address;
      spitch : stddef_h.size_t;
      width : stddef_h.size_t;
      height : stddef_h.size_t;
      kind : udriver_types_h.cudaMemcpyKind;
      stream : udriver_types_h.cudaStream_t) return udriver_types_h.cudaError_t  -- /usr/local/cuda/include//cuda_runtime_api.h:6133
   with Import => True, 
        Convention => C, 
        External_Name => "cudaMemcpy2DToArrayAsync";

   function cudaMemcpy2DFromArrayAsync
     (dst : System.Address;
      dpitch : stddef_h.size_t;
      src : udriver_types_h.cudaArray_const_t;
      wOffset : stddef_h.size_t;
      hOffset : stddef_h.size_t;
      width : stddef_h.size_t;
      height : stddef_h.size_t;
      kind : udriver_types_h.cudaMemcpyKind;
      stream : udriver_types_h.cudaStream_t) return udriver_types_h.cudaError_t  -- /usr/local/cuda/include//cuda_runtime_api.h:6190
   with Import => True, 
        Convention => C, 
        External_Name => "cudaMemcpy2DFromArrayAsync";

   function cudaMemcpyToSymbolAsync
     (symbol : System.Address;
      src : System.Address;
      count : stddef_h.size_t;
      offset : stddef_h.size_t;
      kind : udriver_types_h.cudaMemcpyKind;
      stream : udriver_types_h.cudaStream_t) return udriver_types_h.cudaError_t  -- /usr/local/cuda/include//cuda_runtime_api.h:6241
   with Import => True, 
        Convention => C, 
        External_Name => "cudaMemcpyToSymbolAsync";

   function cudaMemcpyFromSymbolAsync
     (dst : System.Address;
      symbol : System.Address;
      count : stddef_h.size_t;
      offset : stddef_h.size_t;
      kind : udriver_types_h.cudaMemcpyKind;
      stream : udriver_types_h.cudaStream_t) return udriver_types_h.cudaError_t  -- /usr/local/cuda/include//cuda_runtime_api.h:6292
   with Import => True, 
        Convention => C, 
        External_Name => "cudaMemcpyFromSymbolAsync";

   function cudaMemset
     (devPtr : System.Address;
      value : int;
      count : stddef_h.size_t) return udriver_types_h.cudaError_t  -- /usr/local/cuda/include//cuda_runtime_api.h:6321
   with Import => True, 
        Convention => C, 
        External_Name => "cudaMemset";

   function cudaMemset2D
     (devPtr : System.Address;
      pitch : stddef_h.size_t;
      value : int;
      width : stddef_h.size_t;
      height : stddef_h.size_t) return udriver_types_h.cudaError_t  -- /usr/local/cuda/include//cuda_runtime_api.h:6355
   with Import => True, 
        Convention => C, 
        External_Name => "cudaMemset2D";

   function cudaMemset3D
     (pitchedDevPtr : udriver_types_h.cudaPitchedPtr;
      value : int;
      extent : udriver_types_h.cudaExtent) return udriver_types_h.cudaError_t  -- /usr/local/cuda/include//cuda_runtime_api.h:6399
   with Import => True, 
        Convention => C, 
        External_Name => "cudaMemset3D";

   function cudaMemsetAsync
     (devPtr : System.Address;
      value : int;
      count : stddef_h.size_t;
      stream : udriver_types_h.cudaStream_t) return udriver_types_h.cudaError_t  -- /usr/local/cuda/include//cuda_runtime_api.h:6435
   with Import => True, 
        Convention => C, 
        External_Name => "cudaMemsetAsync";

   function cudaMemset2DAsync
     (devPtr : System.Address;
      pitch : stddef_h.size_t;
      value : int;
      width : stddef_h.size_t;
      height : stddef_h.size_t;
      stream : udriver_types_h.cudaStream_t) return udriver_types_h.cudaError_t  -- /usr/local/cuda/include//cuda_runtime_api.h:6476
   with Import => True, 
        Convention => C, 
        External_Name => "cudaMemset2DAsync";

   function cudaMemset3DAsync
     (pitchedDevPtr : udriver_types_h.cudaPitchedPtr;
      value : int;
      extent : udriver_types_h.cudaExtent;
      stream : udriver_types_h.cudaStream_t) return udriver_types_h.cudaError_t  -- /usr/local/cuda/include//cuda_runtime_api.h:6527
   with Import => True, 
        Convention => C, 
        External_Name => "cudaMemset3DAsync";

   function cudaGetSymbolAddress (devPtr : System.Address; symbol : System.Address) return udriver_types_h.cudaError_t  -- /usr/local/cuda/include//cuda_runtime_api.h:6555
   with Import => True, 
        Convention => C, 
        External_Name => "cudaGetSymbolAddress";

   function cudaGetSymbolSize (size : access stddef_h.size_t; symbol : System.Address) return udriver_types_h.cudaError_t  -- /usr/local/cuda/include//cuda_runtime_api.h:6582
   with Import => True, 
        Convention => C, 
        External_Name => "cudaGetSymbolSize";

   function cudaMemPrefetchAsync
     (devPtr : System.Address;
      count : stddef_h.size_t;
      dstDevice : int;
      stream : udriver_types_h.cudaStream_t) return udriver_types_h.cudaError_t  -- /usr/local/cuda/include//cuda_runtime_api.h:6652
   with Import => True, 
        Convention => C, 
        External_Name => "cudaMemPrefetchAsync";

   function cudaMemAdvise
     (devPtr : System.Address;
      count : stddef_h.size_t;
      advice : udriver_types_h.cudaMemoryAdvise;
      device : int) return udriver_types_h.cudaError_t  -- /usr/local/cuda/include//cuda_runtime_api.h:6768
   with Import => True, 
        Convention => C, 
        External_Name => "cudaMemAdvise";

   function cudaMemRangeGetAttribute
     (data : System.Address;
      dataSize : stddef_h.size_t;
      attribute : udriver_types_h.cudaMemRangeAttribute;
      devPtr : System.Address;
      count : stddef_h.size_t) return udriver_types_h.cudaError_t  -- /usr/local/cuda/include//cuda_runtime_api.h:6827
   with Import => True, 
        Convention => C, 
        External_Name => "cudaMemRangeGetAttribute";

   function cudaMemRangeGetAttributes
     (data : System.Address;
      dataSizes : access stddef_h.size_t;
      attributes : access udriver_types_h.cudaMemRangeAttribute;
      numAttributes : stddef_h.size_t;
      devPtr : System.Address;
      count : stddef_h.size_t) return udriver_types_h.cudaError_t  -- /usr/local/cuda/include//cuda_runtime_api.h:6866
   with Import => True, 
        Convention => C, 
        External_Name => "cudaMemRangeGetAttributes";

   function cudaMemcpyToArray
     (dst : udriver_types_h.cudaArray_t;
      wOffset : stddef_h.size_t;
      hOffset : stddef_h.size_t;
      src : System.Address;
      count : stddef_h.size_t;
      kind : udriver_types_h.cudaMemcpyKind) return udriver_types_h.cudaError_t  -- /usr/local/cuda/include//cuda_runtime_api.h:6926
   with Import => True, 
        Convention => C, 
        External_Name => "cudaMemcpyToArray";

   function cudaMemcpyFromArray
     (dst : System.Address;
      src : udriver_types_h.cudaArray_const_t;
      wOffset : stddef_h.size_t;
      hOffset : stddef_h.size_t;
      count : stddef_h.size_t;
      kind : udriver_types_h.cudaMemcpyKind) return udriver_types_h.cudaError_t  -- /usr/local/cuda/include//cuda_runtime_api.h:6968
   with Import => True, 
        Convention => C, 
        External_Name => "cudaMemcpyFromArray";

   function cudaMemcpyArrayToArray
     (dst : udriver_types_h.cudaArray_t;
      wOffsetDst : stddef_h.size_t;
      hOffsetDst : stddef_h.size_t;
      src : udriver_types_h.cudaArray_const_t;
      wOffsetSrc : stddef_h.size_t;
      hOffsetSrc : stddef_h.size_t;
      count : stddef_h.size_t;
      kind : udriver_types_h.cudaMemcpyKind) return udriver_types_h.cudaError_t  -- /usr/local/cuda/include//cuda_runtime_api.h:7011
   with Import => True, 
        Convention => C, 
        External_Name => "cudaMemcpyArrayToArray";

   function cudaMemcpyToArrayAsync
     (dst : udriver_types_h.cudaArray_t;
      wOffset : stddef_h.size_t;
      hOffset : stddef_h.size_t;
      src : System.Address;
      count : stddef_h.size_t;
      kind : udriver_types_h.cudaMemcpyKind;
      stream : udriver_types_h.cudaStream_t) return udriver_types_h.cudaError_t  -- /usr/local/cuda/include//cuda_runtime_api.h:7062
   with Import => True, 
        Convention => C, 
        External_Name => "cudaMemcpyToArrayAsync";

   function cudaMemcpyFromArrayAsync
     (dst : System.Address;
      src : udriver_types_h.cudaArray_const_t;
      wOffset : stddef_h.size_t;
      hOffset : stddef_h.size_t;
      count : stddef_h.size_t;
      kind : udriver_types_h.cudaMemcpyKind;
      stream : udriver_types_h.cudaStream_t) return udriver_types_h.cudaError_t  -- /usr/local/cuda/include//cuda_runtime_api.h:7112
   with Import => True, 
        Convention => C, 
        External_Name => "cudaMemcpyFromArrayAsync";

   function cudaPointerGetAttributes (attributes : access udriver_types_h.cudaPointerAttributes; ptr : System.Address) return udriver_types_h.cudaError_t  -- /usr/local/cuda/include//cuda_runtime_api.h:7267
   with Import => True, 
        Convention => C, 
        External_Name => "cudaPointerGetAttributes";

   function cudaDeviceCanAccessPeer
     (canAccessPeer : access int;
      device : int;
      peerDevice : int) return udriver_types_h.cudaError_t  -- /usr/local/cuda/include//cuda_runtime_api.h:7308
   with Import => True, 
        Convention => C, 
        External_Name => "cudaDeviceCanAccessPeer";

   function cudaDeviceEnablePeerAccess (peerDevice : int; flags : unsigned) return udriver_types_h.cudaError_t  -- /usr/local/cuda/include//cuda_runtime_api.h:7350
   with Import => True, 
        Convention => C, 
        External_Name => "cudaDeviceEnablePeerAccess";

   function cudaDeviceDisablePeerAccess (peerDevice : int) return udriver_types_h.cudaError_t  -- /usr/local/cuda/include//cuda_runtime_api.h:7372
   with Import => True, 
        Convention => C, 
        External_Name => "cudaDeviceDisablePeerAccess";

   function cudaGraphicsUnregisterResource (resource : udriver_types_h.cudaGraphicsResource_t) return udriver_types_h.cudaError_t  -- /usr/local/cuda/include//cuda_runtime_api.h:7435
   with Import => True, 
        Convention => C, 
        External_Name => "cudaGraphicsUnregisterResource";

   function cudaGraphicsResourceSetMapFlags (resource : udriver_types_h.cudaGraphicsResource_t; flags : unsigned) return udriver_types_h.cudaError_t  -- /usr/local/cuda/include//cuda_runtime_api.h:7470
   with Import => True, 
        Convention => C, 
        External_Name => "cudaGraphicsResourceSetMapFlags";

   function cudaGraphicsMapResources
     (count : int;
      resources : System.Address;
      stream : udriver_types_h.cudaStream_t) return udriver_types_h.cudaError_t  -- /usr/local/cuda/include//cuda_runtime_api.h:7509
   with Import => True, 
        Convention => C, 
        External_Name => "cudaGraphicsMapResources";

   function cudaGraphicsUnmapResources
     (count : int;
      resources : System.Address;
      stream : udriver_types_h.cudaStream_t) return udriver_types_h.cudaError_t  -- /usr/local/cuda/include//cuda_runtime_api.h:7544
   with Import => True, 
        Convention => C, 
        External_Name => "cudaGraphicsUnmapResources";

   function cudaGraphicsResourceGetMappedPointer
     (devPtr : System.Address;
      size : access stddef_h.size_t;
      resource : udriver_types_h.cudaGraphicsResource_t) return udriver_types_h.cudaError_t  -- /usr/local/cuda/include//cuda_runtime_api.h:7576
   with Import => True, 
        Convention => C, 
        External_Name => "cudaGraphicsResourceGetMappedPointer";

   function cudaGraphicsSubResourceGetMappedArray
     (c_array : System.Address;
      resource : udriver_types_h.cudaGraphicsResource_t;
      arrayIndex : unsigned;
      mipLevel : unsigned) return udriver_types_h.cudaError_t  -- /usr/local/cuda/include//cuda_runtime_api.h:7614
   with Import => True, 
        Convention => C, 
        External_Name => "cudaGraphicsSubResourceGetMappedArray";

   function cudaGraphicsResourceGetMappedMipmappedArray (mipmappedArray : System.Address; resource : udriver_types_h.cudaGraphicsResource_t) return udriver_types_h.cudaError_t  -- /usr/local/cuda/include//cuda_runtime_api.h:7643
   with Import => True, 
        Convention => C, 
        External_Name => "cudaGraphicsResourceGetMappedMipmappedArray";

   function cudaBindTexture
     (offset : access stddef_h.size_t;
      texref : access constant utexture_types_h.textureReference;
      devPtr : System.Address;
      desc : access constant udriver_types_h.cudaChannelFormatDesc;
      size : stddef_h.size_t) return udriver_types_h.cudaError_t  -- /usr/local/cuda/include//cuda_runtime_api.h:7714
   with Import => True, 
        Convention => C, 
        External_Name => "cudaBindTexture";

   function cudaBindTexture2D
     (offset : access stddef_h.size_t;
      texref : access constant utexture_types_h.textureReference;
      devPtr : System.Address;
      desc : access constant udriver_types_h.cudaChannelFormatDesc;
      width : stddef_h.size_t;
      height : stddef_h.size_t;
      pitch : stddef_h.size_t) return udriver_types_h.cudaError_t  -- /usr/local/cuda/include//cuda_runtime_api.h:7773
   with Import => True, 
        Convention => C, 
        External_Name => "cudaBindTexture2D";

   function cudaBindTextureToArray
     (texref : access constant utexture_types_h.textureReference;
      c_array : udriver_types_h.cudaArray_const_t;
      desc : access constant udriver_types_h.cudaChannelFormatDesc) return udriver_types_h.cudaError_t  -- /usr/local/cuda/include//cuda_runtime_api.h:7811
   with Import => True, 
        Convention => C, 
        External_Name => "cudaBindTextureToArray";

   function cudaBindTextureToMipmappedArray
     (texref : access constant utexture_types_h.textureReference;
      mipmappedArray : udriver_types_h.cudaMipmappedArray_const_t;
      desc : access constant udriver_types_h.cudaChannelFormatDesc) return udriver_types_h.cudaError_t  -- /usr/local/cuda/include//cuda_runtime_api.h:7851
   with Import => True, 
        Convention => C, 
        External_Name => "cudaBindTextureToMipmappedArray";

   function cudaUnbindTexture (texref : access constant utexture_types_h.textureReference) return udriver_types_h.cudaError_t  -- /usr/local/cuda/include//cuda_runtime_api.h:7877
   with Import => True, 
        Convention => C, 
        External_Name => "cudaUnbindTexture";

   function cudaGetTextureAlignmentOffset (offset : access stddef_h.size_t; texref : access constant utexture_types_h.textureReference) return udriver_types_h.cudaError_t  -- /usr/local/cuda/include//cuda_runtime_api.h:7906
   with Import => True, 
        Convention => C, 
        External_Name => "cudaGetTextureAlignmentOffset";

   function cudaGetTextureReference (texref : System.Address; symbol : System.Address) return udriver_types_h.cudaError_t  -- /usr/local/cuda/include//cuda_runtime_api.h:7936
   with Import => True, 
        Convention => C, 
        External_Name => "cudaGetTextureReference";

   function cudaBindSurfaceToArray
     (surfref : access constant usurface_types_h.surfaceReference;
      c_array : udriver_types_h.cudaArray_const_t;
      desc : access constant udriver_types_h.cudaChannelFormatDesc) return udriver_types_h.cudaError_t  -- /usr/local/cuda/include//cuda_runtime_api.h:7981
   with Import => True, 
        Convention => C, 
        External_Name => "cudaBindSurfaceToArray";

   function cudaGetSurfaceReference (surfref : System.Address; symbol : System.Address) return udriver_types_h.cudaError_t  -- /usr/local/cuda/include//cuda_runtime_api.h:8006
   with Import => True, 
        Convention => C, 
        External_Name => "cudaGetSurfaceReference";

   function cudaGetChannelDesc (desc : access udriver_types_h.cudaChannelFormatDesc; c_array : udriver_types_h.cudaArray_const_t) return udriver_types_h.cudaError_t  -- /usr/local/cuda/include//cuda_runtime_api.h:8041
   with Import => True, 
        Convention => C, 
        External_Name => "cudaGetChannelDesc";

   function cudaCreateChannelDesc
     (x : int;
      y : int;
      z : int;
      w : int;
      f : udriver_types_h.cudaChannelFormatKind) return udriver_types_h.cudaChannelFormatDesc  -- /usr/local/cuda/include//cuda_runtime_api.h:8071
   with Import => True, 
        Convention => C, 
        External_Name => "cudaCreateChannelDesc";

   function cudaCreateTextureObject
     (pTexObject : access utexture_types_h.cudaTextureObject_t;
      pResDesc : access constant udriver_types_h.cudaResourceDesc;
      pTexDesc : access constant utexture_types_h.cudaTextureDesc;
      pResViewDesc : access constant udriver_types_h.cudaResourceViewDesc) return udriver_types_h.cudaError_t  -- /usr/local/cuda/include//cuda_runtime_api.h:8289
   with Import => True, 
        Convention => C, 
        External_Name => "cudaCreateTextureObject";

   function cudaDestroyTextureObject (texObject : utexture_types_h.cudaTextureObject_t) return udriver_types_h.cudaError_t  -- /usr/local/cuda/include//cuda_runtime_api.h:8308
   with Import => True, 
        Convention => C, 
        External_Name => "cudaDestroyTextureObject";

   function cudaGetTextureObjectResourceDesc (pResDesc : access udriver_types_h.cudaResourceDesc; texObject : utexture_types_h.cudaTextureObject_t) return udriver_types_h.cudaError_t  -- /usr/local/cuda/include//cuda_runtime_api.h:8328
   with Import => True, 
        Convention => C, 
        External_Name => "cudaGetTextureObjectResourceDesc";

   function cudaGetTextureObjectTextureDesc (pTexDesc : access utexture_types_h.cudaTextureDesc; texObject : utexture_types_h.cudaTextureObject_t) return udriver_types_h.cudaError_t  -- /usr/local/cuda/include//cuda_runtime_api.h:8348
   with Import => True, 
        Convention => C, 
        External_Name => "cudaGetTextureObjectTextureDesc";

   function cudaGetTextureObjectResourceViewDesc (pResViewDesc : access udriver_types_h.cudaResourceViewDesc; texObject : utexture_types_h.cudaTextureObject_t) return udriver_types_h.cudaError_t  -- /usr/local/cuda/include//cuda_runtime_api.h:8369
   with Import => True, 
        Convention => C, 
        External_Name => "cudaGetTextureObjectResourceViewDesc";

   function cudaCreateSurfaceObject (pSurfObject : access usurface_types_h.cudaSurfaceObject_t; pResDesc : access constant udriver_types_h.cudaResourceDesc) return udriver_types_h.cudaError_t  -- /usr/local/cuda/include//cuda_runtime_api.h:8414
   with Import => True, 
        Convention => C, 
        External_Name => "cudaCreateSurfaceObject";

   function cudaDestroySurfaceObject (surfObject : usurface_types_h.cudaSurfaceObject_t) return udriver_types_h.cudaError_t  -- /usr/local/cuda/include//cuda_runtime_api.h:8433
   with Import => True, 
        Convention => C, 
        External_Name => "cudaDestroySurfaceObject";

   function cudaGetSurfaceObjectResourceDesc (pResDesc : access udriver_types_h.cudaResourceDesc; surfObject : usurface_types_h.cudaSurfaceObject_t) return udriver_types_h.cudaError_t  -- /usr/local/cuda/include//cuda_runtime_api.h:8452
   with Import => True, 
        Convention => C, 
        External_Name => "cudaGetSurfaceObjectResourceDesc";

   function cudaDriverGetVersion (driverVersion : access int) return udriver_types_h.cudaError_t  -- /usr/local/cuda/include//cuda_runtime_api.h:8486
   with Import => True, 
        Convention => C, 
        External_Name => "cudaDriverGetVersion";

   function cudaRuntimeGetVersion (runtimeVersion : access int) return udriver_types_h.cudaError_t  -- /usr/local/cuda/include//cuda_runtime_api.h:8511
   with Import => True, 
        Convention => C, 
        External_Name => "cudaRuntimeGetVersion";

   function cudaGraphCreate (pGraph : System.Address; flags : unsigned) return udriver_types_h.cudaError_t  -- /usr/local/cuda/include//cuda_runtime_api.h:8558
   with Import => True, 
        Convention => C, 
        External_Name => "cudaGraphCreate";

   function cudaGraphAddKernelNode
     (pGraphNode : System.Address;
      graph : udriver_types_h.cudaGraph_t;
      pDependencies : System.Address;
      numDependencies : stddef_h.size_t;
      pNodeParams : access constant udriver_types_h.cudaKernelNodeParams) return udriver_types_h.cudaError_t  -- /usr/local/cuda/include//cuda_runtime_api.h:8655
   with Import => True, 
        Convention => C, 
        External_Name => "cudaGraphAddKernelNode";

   function cudaGraphKernelNodeGetParams (node : udriver_types_h.cudaGraphNode_t; pNodeParams : access udriver_types_h.cudaKernelNodeParams) return udriver_types_h.cudaError_t  -- /usr/local/cuda/include//cuda_runtime_api.h:8688
   with Import => True, 
        Convention => C, 
        External_Name => "cudaGraphKernelNodeGetParams";

   function cudaGraphKernelNodeSetParams (node : udriver_types_h.cudaGraphNode_t; pNodeParams : access constant udriver_types_h.cudaKernelNodeParams) return udriver_types_h.cudaError_t  -- /usr/local/cuda/include//cuda_runtime_api.h:8713
   with Import => True, 
        Convention => C, 
        External_Name => "cudaGraphKernelNodeSetParams";

   function cudaGraphKernelNodeCopyAttributes (hSrc : udriver_types_h.cudaGraphNode_t; hDst : udriver_types_h.cudaGraphNode_t) return udriver_types_h.cudaError_t  -- /usr/local/cuda/include//cuda_runtime_api.h:8733
   with Import => True, 
        Convention => C, 
        External_Name => "cudaGraphKernelNodeCopyAttributes";

   function cudaGraphKernelNodeGetAttribute
     (hNode : udriver_types_h.cudaGraphNode_t;
      attr : udriver_types_h.cudaKernelNodeAttrID;
      value_out : access udriver_types_h.cudaKernelNodeAttrValue) return udriver_types_h.cudaError_t  -- /usr/local/cuda/include//cuda_runtime_api.h:8756
   with Import => True, 
        Convention => C, 
        External_Name => "cudaGraphKernelNodeGetAttribute";

   function cudaGraphKernelNodeSetAttribute
     (hNode : udriver_types_h.cudaGraphNode_t;
      attr : udriver_types_h.cudaKernelNodeAttrID;
      value : access constant udriver_types_h.cudaKernelNodeAttrValue) return udriver_types_h.cudaError_t  -- /usr/local/cuda/include//cuda_runtime_api.h:8780
   with Import => True, 
        Convention => C, 
        External_Name => "cudaGraphKernelNodeSetAttribute";

   function cudaGraphAddMemcpyNode
     (pGraphNode : System.Address;
      graph : udriver_types_h.cudaGraph_t;
      pDependencies : System.Address;
      numDependencies : stddef_h.size_t;
      pCopyParams : access constant udriver_types_h.cudaMemcpy3DParms) return udriver_types_h.cudaError_t  -- /usr/local/cuda/include//cuda_runtime_api.h:8827
   with Import => True, 
        Convention => C, 
        External_Name => "cudaGraphAddMemcpyNode";

   function cudaGraphMemcpyNodeGetParams (node : udriver_types_h.cudaGraphNode_t; pNodeParams : access udriver_types_h.cudaMemcpy3DParms) return udriver_types_h.cudaError_t  -- /usr/local/cuda/include//cuda_runtime_api.h:8850
   with Import => True, 
        Convention => C, 
        External_Name => "cudaGraphMemcpyNodeGetParams";

   function cudaGraphMemcpyNodeSetParams (node : udriver_types_h.cudaGraphNode_t; pNodeParams : access constant udriver_types_h.cudaMemcpy3DParms) return udriver_types_h.cudaError_t  -- /usr/local/cuda/include//cuda_runtime_api.h:8873
   with Import => True, 
        Convention => C, 
        External_Name => "cudaGraphMemcpyNodeSetParams";

   function cudaGraphAddMemsetNode
     (pGraphNode : System.Address;
      graph : udriver_types_h.cudaGraph_t;
      pDependencies : System.Address;
      numDependencies : stddef_h.size_t;
      pMemsetParams : access constant udriver_types_h.cudaMemsetParams) return udriver_types_h.cudaError_t  -- /usr/local/cuda/include//cuda_runtime_api.h:8915
   with Import => True, 
        Convention => C, 
        External_Name => "cudaGraphAddMemsetNode";

   function cudaGraphMemsetNodeGetParams (node : udriver_types_h.cudaGraphNode_t; pNodeParams : access udriver_types_h.cudaMemsetParams) return udriver_types_h.cudaError_t  -- /usr/local/cuda/include//cuda_runtime_api.h:8938
   with Import => True, 
        Convention => C, 
        External_Name => "cudaGraphMemsetNodeGetParams";

   function cudaGraphMemsetNodeSetParams (node : udriver_types_h.cudaGraphNode_t; pNodeParams : access constant udriver_types_h.cudaMemsetParams) return udriver_types_h.cudaError_t  -- /usr/local/cuda/include//cuda_runtime_api.h:8961
   with Import => True, 
        Convention => C, 
        External_Name => "cudaGraphMemsetNodeSetParams";

   function cudaGraphAddHostNode
     (pGraphNode : System.Address;
      graph : udriver_types_h.cudaGraph_t;
      pDependencies : System.Address;
      numDependencies : stddef_h.size_t;
      pNodeParams : access constant udriver_types_h.cudaHostNodeParams) return udriver_types_h.cudaError_t  -- /usr/local/cuda/include//cuda_runtime_api.h:9002
   with Import => True, 
        Convention => C, 
        External_Name => "cudaGraphAddHostNode";

   function cudaGraphHostNodeGetParams (node : udriver_types_h.cudaGraphNode_t; pNodeParams : access udriver_types_h.cudaHostNodeParams) return udriver_types_h.cudaError_t  -- /usr/local/cuda/include//cuda_runtime_api.h:9025
   with Import => True, 
        Convention => C, 
        External_Name => "cudaGraphHostNodeGetParams";

   function cudaGraphHostNodeSetParams (node : udriver_types_h.cudaGraphNode_t; pNodeParams : access constant udriver_types_h.cudaHostNodeParams) return udriver_types_h.cudaError_t  -- /usr/local/cuda/include//cuda_runtime_api.h:9048
   with Import => True, 
        Convention => C, 
        External_Name => "cudaGraphHostNodeSetParams";

   function cudaGraphAddChildGraphNode
     (pGraphNode : System.Address;
      graph : udriver_types_h.cudaGraph_t;
      pDependencies : System.Address;
      numDependencies : stddef_h.size_t;
      childGraph : udriver_types_h.cudaGraph_t) return udriver_types_h.cudaError_t  -- /usr/local/cuda/include//cuda_runtime_api.h:9086
   with Import => True, 
        Convention => C, 
        External_Name => "cudaGraphAddChildGraphNode";

   function cudaGraphChildGraphNodeGetGraph (node : udriver_types_h.cudaGraphNode_t; pGraph : System.Address) return udriver_types_h.cudaError_t  -- /usr/local/cuda/include//cuda_runtime_api.h:9110
   with Import => True, 
        Convention => C, 
        External_Name => "cudaGraphChildGraphNodeGetGraph";

   function cudaGraphAddEmptyNode
     (pGraphNode : System.Address;
      graph : udriver_types_h.cudaGraph_t;
      pDependencies : System.Address;
      numDependencies : stddef_h.size_t) return udriver_types_h.cudaError_t  -- /usr/local/cuda/include//cuda_runtime_api.h:9147
   with Import => True, 
        Convention => C, 
        External_Name => "cudaGraphAddEmptyNode";

   function cudaGraphClone (pGraphClone : System.Address; originalGraph : udriver_types_h.cudaGraph_t) return udriver_types_h.cudaError_t  -- /usr/local/cuda/include//cuda_runtime_api.h:9174
   with Import => True, 
        Convention => C, 
        External_Name => "cudaGraphClone";

   function cudaGraphNodeFindInClone
     (pNode : System.Address;
      originalNode : udriver_types_h.cudaGraphNode_t;
      clonedGraph : udriver_types_h.cudaGraph_t) return udriver_types_h.cudaError_t  -- /usr/local/cuda/include//cuda_runtime_api.h:9202
   with Import => True, 
        Convention => C, 
        External_Name => "cudaGraphNodeFindInClone";

   function cudaGraphNodeGetType (node : udriver_types_h.cudaGraphNode_t; pType : access udriver_types_h.cudaGraphNodeType) return udriver_types_h.cudaError_t  -- /usr/local/cuda/include//cuda_runtime_api.h:9233
   with Import => True, 
        Convention => C, 
        External_Name => "cudaGraphNodeGetType";

   function cudaGraphGetNodes
     (graph : udriver_types_h.cudaGraph_t;
      nodes : System.Address;
      numNodes : access stddef_h.size_t) return udriver_types_h.cudaError_t  -- /usr/local/cuda/include//cuda_runtime_api.h:9264
   with Import => True, 
        Convention => C, 
        External_Name => "cudaGraphGetNodes";

   function cudaGraphGetRootNodes
     (graph : udriver_types_h.cudaGraph_t;
      pRootNodes : System.Address;
      pNumRootNodes : access stddef_h.size_t) return udriver_types_h.cudaError_t  -- /usr/local/cuda/include//cuda_runtime_api.h:9295
   with Import => True, 
        Convention => C, 
        External_Name => "cudaGraphGetRootNodes";

   function cudaGraphGetEdges
     (graph : udriver_types_h.cudaGraph_t;
      from : System.Address;
      to : System.Address;
      numEdges : access stddef_h.size_t) return udriver_types_h.cudaError_t  -- /usr/local/cuda/include//cuda_runtime_api.h:9329
   with Import => True, 
        Convention => C, 
        External_Name => "cudaGraphGetEdges";

   function cudaGraphNodeGetDependencies
     (node : udriver_types_h.cudaGraphNode_t;
      pDependencies : System.Address;
      pNumDependencies : access stddef_h.size_t) return udriver_types_h.cudaError_t  -- /usr/local/cuda/include//cuda_runtime_api.h:9360
   with Import => True, 
        Convention => C, 
        External_Name => "cudaGraphNodeGetDependencies";

   function cudaGraphNodeGetDependentNodes
     (node : udriver_types_h.cudaGraphNode_t;
      pDependentNodes : System.Address;
      pNumDependentNodes : access stddef_h.size_t) return udriver_types_h.cudaError_t  -- /usr/local/cuda/include//cuda_runtime_api.h:9392
   with Import => True, 
        Convention => C, 
        External_Name => "cudaGraphNodeGetDependentNodes";

   function cudaGraphAddDependencies
     (graph : udriver_types_h.cudaGraph_t;
      from : System.Address;
      to : System.Address;
      numDependencies : stddef_h.size_t) return udriver_types_h.cudaError_t  -- /usr/local/cuda/include//cuda_runtime_api.h:9423
   with Import => True, 
        Convention => C, 
        External_Name => "cudaGraphAddDependencies";

   function cudaGraphRemoveDependencies
     (graph : udriver_types_h.cudaGraph_t;
      from : System.Address;
      to : System.Address;
      numDependencies : stddef_h.size_t) return udriver_types_h.cudaError_t  -- /usr/local/cuda/include//cuda_runtime_api.h:9454
   with Import => True, 
        Convention => C, 
        External_Name => "cudaGraphRemoveDependencies";

   function cudaGraphDestroyNode (node : udriver_types_h.cudaGraphNode_t) return udriver_types_h.cudaError_t  -- /usr/local/cuda/include//cuda_runtime_api.h:9480
   with Import => True, 
        Convention => C, 
        External_Name => "cudaGraphDestroyNode";

   function cudaGraphInstantiate
     (pGraphExec : System.Address;
      graph : udriver_types_h.cudaGraph_t;
      pErrorNode : System.Address;
      pLogBuffer : Interfaces.C.Strings.chars_ptr;
      bufferSize : stddef_h.size_t) return udriver_types_h.cudaError_t  -- /usr/local/cuda/include//cuda_runtime_api.h:9516
   with Import => True, 
        Convention => C, 
        External_Name => "cudaGraphInstantiate";

   function cudaGraphExecKernelNodeSetParams
     (hGraphExec : udriver_types_h.cudaGraphExec_t;
      node : udriver_types_h.cudaGraphNode_t;
      pNodeParams : access constant udriver_types_h.cudaKernelNodeParams) return udriver_types_h.cudaError_t  -- /usr/local/cuda/include//cuda_runtime_api.h:9550
   with Import => True, 
        Convention => C, 
        External_Name => "cudaGraphExecKernelNodeSetParams";

   function cudaGraphExecMemcpyNodeSetParams
     (hGraphExec : udriver_types_h.cudaGraphExec_t;
      node : udriver_types_h.cudaGraphNode_t;
      pNodeParams : access constant udriver_types_h.cudaMemcpy3DParms) return udriver_types_h.cudaError_t  -- /usr/local/cuda/include//cuda_runtime_api.h:9591
   with Import => True, 
        Convention => C, 
        External_Name => "cudaGraphExecMemcpyNodeSetParams";

   function cudaGraphExecMemsetNodeSetParams
     (hGraphExec : udriver_types_h.cudaGraphExec_t;
      node : udriver_types_h.cudaGraphNode_t;
      pNodeParams : access constant udriver_types_h.cudaMemsetParams) return udriver_types_h.cudaError_t  -- /usr/local/cuda/include//cuda_runtime_api.h:9632
   with Import => True, 
        Convention => C, 
        External_Name => "cudaGraphExecMemsetNodeSetParams";

   function cudaGraphExecHostNodeSetParams
     (hGraphExec : udriver_types_h.cudaGraphExec_t;
      node : udriver_types_h.cudaGraphNode_t;
      pNodeParams : access constant udriver_types_h.cudaHostNodeParams) return udriver_types_h.cudaError_t  -- /usr/local/cuda/include//cuda_runtime_api.h:9665
   with Import => True, 
        Convention => C, 
        External_Name => "cudaGraphExecHostNodeSetParams";

   function cudaGraphExecUpdate
     (hGraphExec : udriver_types_h.cudaGraphExec_t;
      hGraph : udriver_types_h.cudaGraph_t;
      hErrorNode_out : System.Address;
      updateResult_out : access udriver_types_h.cudaGraphExecUpdateResult) return udriver_types_h.cudaError_t  -- /usr/local/cuda/include//cuda_runtime_api.h:9740
   with Import => True, 
        Convention => C, 
        External_Name => "cudaGraphExecUpdate";

   function cudaGraphLaunch (graphExec : udriver_types_h.cudaGraphExec_t; stream : udriver_types_h.cudaStream_t) return udriver_types_h.cudaError_t  -- /usr/local/cuda/include//cuda_runtime_api.h:9765
   with Import => True, 
        Convention => C, 
        External_Name => "cudaGraphLaunch";

   function cudaGraphExecDestroy (graphExec : udriver_types_h.cudaGraphExec_t) return udriver_types_h.cudaError_t  -- /usr/local/cuda/include//cuda_runtime_api.h:9786
   with Import => True, 
        Convention => C, 
        External_Name => "cudaGraphExecDestroy";

   function cudaGraphDestroy (graph : udriver_types_h.cudaGraph_t) return udriver_types_h.cudaError_t  -- /usr/local/cuda/include//cuda_runtime_api.h:9806
   with Import => True, 
        Convention => C, 
        External_Name => "cudaGraphDestroy";

   function cudaGetExportTable (ppExportTable : System.Address; pExportTableId : access constant udriver_types_h.cudaUUID_t) return udriver_types_h.cudaError_t  -- /usr/local/cuda/include//cuda_runtime_api.h:9811
   with Import => True, 
        Convention => C, 
        External_Name => "cudaGetExportTable";

   function cudaGetFuncBySymbol (functionPtr : System.Address; symbolPtr : System.Address) return udriver_types_h.cudaError_t  -- /usr/local/cuda/include//cuda_runtime_api.h:9987
   with Import => True, 
        Convention => C, 
        External_Name => "cudaGetFuncBySymbol";

end ucuda_runtime_api_h;
