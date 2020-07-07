pragma Ada_2012;
pragma Style_Checks (Off);
pragma Warnings ("U");

with Interfaces.C; use Interfaces.C;
with driver_types_h;
with corecrt_h;
with Interfaces.C.Strings;
with System;
with Interfaces.C.Extensions;
with vector_types_h;
with texture_types_h;
with surface_types_h;

package cuda_runtime_api_h is

   CUDART_VERSION : constant := 10020;  --  /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/cuda_runtime_api.h:136
   --  unsupported macro: CUDART_DEVICE __device__

   function cudaDeviceReset return driver_types_h.cudaError_t  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/cuda_runtime_api.h:280
   with Import => True, 
        Convention => C, 
        External_Name => "cudaDeviceReset";

   function cudaDeviceSynchronize return driver_types_h.cudaError_t  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/cuda_runtime_api.h:301
   with Import => True, 
        Convention => C, 
        External_Name => "cudaDeviceSynchronize";

   function cudaDeviceSetLimit (limit : driver_types_h.cudaLimit; value : corecrt_h.size_t) return driver_types_h.cudaError_t  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/cuda_runtime_api.h:386
   with Import => True, 
        Convention => C, 
        External_Name => "cudaDeviceSetLimit";

   function cudaDeviceGetLimit (pValue : access corecrt_h.size_t; limit : driver_types_h.cudaLimit) return driver_types_h.cudaError_t  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/cuda_runtime_api.h:420
   with Import => True, 
        Convention => C, 
        External_Name => "cudaDeviceGetLimit";

   function cudaDeviceGetCacheConfig (pCacheConfig : access driver_types_h.cudaFuncCache) return driver_types_h.cudaError_t  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/cuda_runtime_api.h:453
   with Import => True, 
        Convention => C, 
        External_Name => "cudaDeviceGetCacheConfig";

   function cudaDeviceGetStreamPriorityRange (leastPriority : access int; greatestPriority : access int) return driver_types_h.cudaError_t  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/cuda_runtime_api.h:490
   with Import => True, 
        Convention => C, 
        External_Name => "cudaDeviceGetStreamPriorityRange";

   function cudaDeviceSetCacheConfig (cacheConfig : driver_types_h.cudaFuncCache) return driver_types_h.cudaError_t  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/cuda_runtime_api.h:534
   with Import => True, 
        Convention => C, 
        External_Name => "cudaDeviceSetCacheConfig";

   function cudaDeviceGetSharedMemConfig (pConfig : access driver_types_h.cudaSharedMemConfig) return driver_types_h.cudaError_t  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/cuda_runtime_api.h:565
   with Import => True, 
        Convention => C, 
        External_Name => "cudaDeviceGetSharedMemConfig";

   function cudaDeviceSetSharedMemConfig (config : driver_types_h.cudaSharedMemConfig) return driver_types_h.cudaError_t  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/cuda_runtime_api.h:609
   with Import => True, 
        Convention => C, 
        External_Name => "cudaDeviceSetSharedMemConfig";

   function cudaDeviceGetByPCIBusId (device : access int; pciBusId : Interfaces.C.Strings.chars_ptr) return driver_types_h.cudaError_t  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/cuda_runtime_api.h:636
   with Import => True, 
        Convention => C, 
        External_Name => "cudaDeviceGetByPCIBusId";

   function cudaDeviceGetPCIBusId
     (pciBusId : Interfaces.C.Strings.chars_ptr;
      len : int;
      device : int) return driver_types_h.cudaError_t  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/cuda_runtime_api.h:666
   with Import => True, 
        Convention => C, 
        External_Name => "cudaDeviceGetPCIBusId";

   function cudaIpcGetEventHandle (handle : access driver_types_h.cudaIpcEventHandle_t; event : driver_types_h.cudaEvent_t) return driver_types_h.cudaError_t  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/cuda_runtime_api.h:713
   with Import => True, 
        Convention => C, 
        External_Name => "cudaIpcGetEventHandle";

   function cudaIpcOpenEventHandle (event : System.Address; handle : driver_types_h.cudaIpcEventHandle_t) return driver_types_h.cudaError_t  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/cuda_runtime_api.h:753
   with Import => True, 
        Convention => C, 
        External_Name => "cudaIpcOpenEventHandle";

   function cudaIpcGetMemHandle (handle : access driver_types_h.cudaIpcMemHandle_t; devPtr : System.Address) return driver_types_h.cudaError_t  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/cuda_runtime_api.h:796
   with Import => True, 
        Convention => C, 
        External_Name => "cudaIpcGetMemHandle";

   function cudaIpcOpenMemHandle
     (devPtr : System.Address;
      handle : driver_types_h.cudaIpcMemHandle_t;
      flags : unsigned) return driver_types_h.cudaError_t  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/cuda_runtime_api.h:854
   with Import => True, 
        Convention => C, 
        External_Name => "cudaIpcOpenMemHandle";

   function cudaIpcCloseMemHandle (devPtr : System.Address) return driver_types_h.cudaError_t  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/cuda_runtime_api.h:889
   with Import => True, 
        Convention => C, 
        External_Name => "cudaIpcCloseMemHandle";

   function cudaThreadExit return driver_types_h.cudaError_t  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/cuda_runtime_api.h:931
   with Import => True, 
        Convention => C, 
        External_Name => "cudaThreadExit";

   function cudaThreadSynchronize return driver_types_h.cudaError_t  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/cuda_runtime_api.h:957
   with Import => True, 
        Convention => C, 
        External_Name => "cudaThreadSynchronize";

   function cudaThreadSetLimit (limit : driver_types_h.cudaLimit; value : corecrt_h.size_t) return driver_types_h.cudaError_t  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/cuda_runtime_api.h:1006
   with Import => True, 
        Convention => C, 
        External_Name => "cudaThreadSetLimit";

   function cudaThreadGetLimit (pValue : access corecrt_h.size_t; limit : driver_types_h.cudaLimit) return driver_types_h.cudaError_t  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/cuda_runtime_api.h:1039
   with Import => True, 
        Convention => C, 
        External_Name => "cudaThreadGetLimit";

   function cudaThreadGetCacheConfig (pCacheConfig : access driver_types_h.cudaFuncCache) return driver_types_h.cudaError_t  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/cuda_runtime_api.h:1075
   with Import => True, 
        Convention => C, 
        External_Name => "cudaThreadGetCacheConfig";

   function cudaThreadSetCacheConfig (cacheConfig : driver_types_h.cudaFuncCache) return driver_types_h.cudaError_t  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/cuda_runtime_api.h:1122
   with Import => True, 
        Convention => C, 
        External_Name => "cudaThreadSetCacheConfig";

   function cudaGetLastError return driver_types_h.cudaError_t  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/cuda_runtime_api.h:1181
   with Import => True, 
        Convention => C, 
        External_Name => "cudaGetLastError";

   function cudaPeekAtLastError return driver_types_h.cudaError_t  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/cuda_runtime_api.h:1227
   with Import => True, 
        Convention => C, 
        External_Name => "cudaPeekAtLastError";

   function cudaGetErrorName (arg1 : driver_types_h.cudaError_t) return Interfaces.C.Strings.chars_ptr  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/cuda_runtime_api.h:1243
   with Import => True, 
        Convention => C, 
        External_Name => "cudaGetErrorName";

   function cudaGetErrorString (arg1 : driver_types_h.cudaError_t) return Interfaces.C.Strings.chars_ptr  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/cuda_runtime_api.h:1259
   with Import => True, 
        Convention => C, 
        External_Name => "cudaGetErrorString";

   function cudaGetDeviceCount (count : access int) return driver_types_h.cudaError_t  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/cuda_runtime_api.h:1288
   with Import => True, 
        Convention => C, 
        External_Name => "cudaGetDeviceCount";

   function cudaGetDeviceProperties (prop : access driver_types_h.cudaDeviceProp; device : int) return driver_types_h.cudaError_t  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/cuda_runtime_api.h:1559
   with Import => True, 
        Convention => C, 
        External_Name => "cudaGetDeviceProperties";

   function cudaDeviceGetAttribute
     (value : access int;
      attr : driver_types_h.cudaDeviceAttr;
      device : int) return driver_types_h.cudaError_t  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/cuda_runtime_api.h:1748
   with Import => True, 
        Convention => C, 
        External_Name => "cudaDeviceGetAttribute";

   function cudaDeviceGetNvSciSyncAttributes
     (nvSciSyncAttrList : System.Address;
      device : int;
      flags : int) return driver_types_h.cudaError_t  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/cuda_runtime_api.h:1797
   with Import => True, 
        Convention => C, 
        External_Name => "cudaDeviceGetNvSciSyncAttributes";

   function cudaDeviceGetP2PAttribute
     (value : access int;
      attr : driver_types_h.cudaDeviceP2PAttr;
      srcDevice : int;
      dstDevice : int) return driver_types_h.cudaError_t  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/cuda_runtime_api.h:1837
   with Import => True, 
        Convention => C, 
        External_Name => "cudaDeviceGetP2PAttribute";

   function cudaChooseDevice (device : access int; prop : access constant driver_types_h.cudaDeviceProp) return driver_types_h.cudaError_t  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/cuda_runtime_api.h:1858
   with Import => True, 
        Convention => C, 
        External_Name => "cudaChooseDevice";

   function cudaSetDevice (device : int) return driver_types_h.cudaError_t  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/cuda_runtime_api.h:1895
   with Import => True, 
        Convention => C, 
        External_Name => "cudaSetDevice";

   function cudaGetDevice (device : access int) return driver_types_h.cudaError_t  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/cuda_runtime_api.h:1916
   with Import => True, 
        Convention => C, 
        External_Name => "cudaGetDevice";

   function cudaSetValidDevices (device_arr : access int; len : int) return driver_types_h.cudaError_t  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/cuda_runtime_api.h:1947
   with Import => True, 
        Convention => C, 
        External_Name => "cudaSetValidDevices";

   function cudaSetDeviceFlags (flags : unsigned) return driver_types_h.cudaError_t  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/cuda_runtime_api.h:2016
   with Import => True, 
        Convention => C, 
        External_Name => "cudaSetDeviceFlags";

   function cudaGetDeviceFlags (flags : access unsigned) return driver_types_h.cudaError_t  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/cuda_runtime_api.h:2062
   with Import => True, 
        Convention => C, 
        External_Name => "cudaGetDeviceFlags";

   function cudaStreamCreate (pStream : System.Address) return driver_types_h.cudaError_t  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/cuda_runtime_api.h:2102
   with Import => True, 
        Convention => C, 
        External_Name => "cudaStreamCreate";

   function cudaStreamCreateWithFlags (pStream : System.Address; flags : unsigned) return driver_types_h.cudaError_t  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/cuda_runtime_api.h:2134
   with Import => True, 
        Convention => C, 
        External_Name => "cudaStreamCreateWithFlags";

   function cudaStreamCreateWithPriority
     (pStream : System.Address;
      flags : unsigned;
      priority : int) return driver_types_h.cudaError_t  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/cuda_runtime_api.h:2180
   with Import => True, 
        Convention => C, 
        External_Name => "cudaStreamCreateWithPriority";

   function cudaStreamGetPriority (hStream : driver_types_h.cudaStream_t; priority : access int) return driver_types_h.cudaError_t  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/cuda_runtime_api.h:2207
   with Import => True, 
        Convention => C, 
        External_Name => "cudaStreamGetPriority";

   function cudaStreamGetFlags (hStream : driver_types_h.cudaStream_t; flags : access unsigned) return driver_types_h.cudaError_t  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/cuda_runtime_api.h:2232
   with Import => True, 
        Convention => C, 
        External_Name => "cudaStreamGetFlags";

   function cudaStreamDestroy (stream : driver_types_h.cudaStream_t) return driver_types_h.cudaError_t  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/cuda_runtime_api.h:2263
   with Import => True, 
        Convention => C, 
        External_Name => "cudaStreamDestroy";

   function cudaStreamWaitEvent
     (stream : driver_types_h.cudaStream_t;
      event : driver_types_h.cudaEvent_t;
      flags : unsigned) return driver_types_h.cudaError_t  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/cuda_runtime_api.h:2289
   with Import => True, 
        Convention => C, 
        External_Name => "cudaStreamWaitEvent";

   type cudaStreamCallback_t is access procedure
        (arg1 : driver_types_h.cudaStream_t;
         arg2 : driver_types_h.cudaError_t;
         arg3 : System.Address)
   with Convention => C;  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/cuda_runtime_api.h:2297

   function cudaStreamAddCallback
     (stream : driver_types_h.cudaStream_t;
      callback : cudaStreamCallback_t;
      userData : System.Address;
      flags : unsigned) return driver_types_h.cudaError_t  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/cuda_runtime_api.h:2364
   with Import => True, 
        Convention => C, 
        External_Name => "cudaStreamAddCallback";

   function cudaStreamSynchronize (stream : driver_types_h.cudaStream_t) return driver_types_h.cudaError_t  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/cuda_runtime_api.h:2388
   with Import => True, 
        Convention => C, 
        External_Name => "cudaStreamSynchronize";

   function cudaStreamQuery (stream : driver_types_h.cudaStream_t) return driver_types_h.cudaError_t  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/cuda_runtime_api.h:2413
   with Import => True, 
        Convention => C, 
        External_Name => "cudaStreamQuery";

   function cudaStreamAttachMemAsync
     (stream : driver_types_h.cudaStream_t;
      devPtr : System.Address;
      length : corecrt_h.size_t;
      flags : unsigned) return driver_types_h.cudaError_t  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/cuda_runtime_api.h:2496
   with Import => True, 
        Convention => C, 
        External_Name => "cudaStreamAttachMemAsync";

   function cudaStreamBeginCapture (stream : driver_types_h.cudaStream_t; mode : driver_types_h.cudaStreamCaptureMode) return driver_types_h.cudaError_t  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/cuda_runtime_api.h:2532
   with Import => True, 
        Convention => C, 
        External_Name => "cudaStreamBeginCapture";

   function cudaThreadExchangeStreamCaptureMode (mode : access driver_types_h.cudaStreamCaptureMode) return driver_types_h.cudaError_t  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/cuda_runtime_api.h:2583
   with Import => True, 
        Convention => C, 
        External_Name => "cudaThreadExchangeStreamCaptureMode";

   function cudaStreamEndCapture (stream : driver_types_h.cudaStream_t; pGraph : System.Address) return driver_types_h.cudaError_t  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/cuda_runtime_api.h:2611
   with Import => True, 
        Convention => C, 
        External_Name => "cudaStreamEndCapture";

   function cudaStreamIsCapturing (stream : driver_types_h.cudaStream_t; pCaptureStatus : access driver_types_h.cudaStreamCaptureStatus) return driver_types_h.cudaError_t  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/cuda_runtime_api.h:2649
   with Import => True, 
        Convention => C, 
        External_Name => "cudaStreamIsCapturing";

   function cudaStreamGetCaptureInfo
     (stream : driver_types_h.cudaStream_t;
      pCaptureStatus : access driver_types_h.cudaStreamCaptureStatus;
      pId : access Extensions.unsigned_long_long) return driver_types_h.cudaError_t  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/cuda_runtime_api.h:2677
   with Import => True, 
        Convention => C, 
        External_Name => "cudaStreamGetCaptureInfo";

   function cudaEventCreate (event : System.Address) return driver_types_h.cudaError_t  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/cuda_runtime_api.h:2714
   with Import => True, 
        Convention => C, 
        External_Name => "cudaEventCreate";

   function cudaEventCreateWithFlags (event : System.Address; flags : unsigned) return driver_types_h.cudaError_t  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/cuda_runtime_api.h:2751
   with Import => True, 
        Convention => C, 
        External_Name => "cudaEventCreateWithFlags";

   function cudaEventRecord (event : driver_types_h.cudaEvent_t; stream : driver_types_h.cudaStream_t) return driver_types_h.cudaError_t  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/cuda_runtime_api.h:2790
   with Import => True, 
        Convention => C, 
        External_Name => "cudaEventRecord";

   function cudaEventQuery (event : driver_types_h.cudaEvent_t) return driver_types_h.cudaError_t  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/cuda_runtime_api.h:2821
   with Import => True, 
        Convention => C, 
        External_Name => "cudaEventQuery";

   function cudaEventSynchronize (event : driver_types_h.cudaEvent_t) return driver_types_h.cudaError_t  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/cuda_runtime_api.h:2851
   with Import => True, 
        Convention => C, 
        External_Name => "cudaEventSynchronize";

   function cudaEventDestroy (event : driver_types_h.cudaEvent_t) return driver_types_h.cudaError_t  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/cuda_runtime_api.h:2878
   with Import => True, 
        Convention => C, 
        External_Name => "cudaEventDestroy";

   function cudaEventElapsedTime
     (ms : access float;
      start : driver_types_h.cudaEvent_t;
      c_end : driver_types_h.cudaEvent_t) return driver_types_h.cudaError_t  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/cuda_runtime_api.h:2921
   with Import => True, 
        Convention => C, 
        External_Name => "cudaEventElapsedTime";

   function cudaImportExternalMemory (extMem_out : System.Address; memHandleDesc : access constant driver_types_h.cudaExternalMemoryHandleDesc) return driver_types_h.cudaError_t  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/cuda_runtime_api.h:3098
   with Import => True, 
        Convention => C, 
        External_Name => "cudaImportExternalMemory";

   function cudaExternalMemoryGetMappedBuffer
     (devPtr : System.Address;
      extMem : driver_types_h.cudaExternalMemory_t;
      bufferDesc : access constant driver_types_h.cudaExternalMemoryBufferDesc) return driver_types_h.cudaError_t  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/cuda_runtime_api.h:3152
   with Import => True, 
        Convention => C, 
        External_Name => "cudaExternalMemoryGetMappedBuffer";

   function cudaExternalMemoryGetMappedMipmappedArray
     (mipmap : System.Address;
      extMem : driver_types_h.cudaExternalMemory_t;
      mipmapDesc : access constant driver_types_h.cudaExternalMemoryMipmappedArrayDesc) return driver_types_h.cudaError_t  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/cuda_runtime_api.h:3211
   with Import => True, 
        Convention => C, 
        External_Name => "cudaExternalMemoryGetMappedMipmappedArray";

   function cudaDestroyExternalMemory (extMem : driver_types_h.cudaExternalMemory_t) return driver_types_h.cudaError_t  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/cuda_runtime_api.h:3234
   with Import => True, 
        Convention => C, 
        External_Name => "cudaDestroyExternalMemory";

   function cudaImportExternalSemaphore (extSem_out : System.Address; semHandleDesc : access constant driver_types_h.cudaExternalSemaphoreHandleDesc) return driver_types_h.cudaError_t  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/cuda_runtime_api.h:3365
   with Import => True, 
        Convention => C, 
        External_Name => "cudaImportExternalSemaphore";

   function cudaSignalExternalSemaphoresAsync
     (extSemArray : System.Address;
      paramsArray : access constant driver_types_h.cudaExternalSemaphoreSignalParams;
      numExtSems : unsigned;
      stream : driver_types_h.cudaStream_t) return driver_types_h.cudaError_t  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/cuda_runtime_api.h:3430
   with Import => True, 
        Convention => C, 
        External_Name => "cudaSignalExternalSemaphoresAsync";

   function cudaWaitExternalSemaphoresAsync
     (extSemArray : System.Address;
      paramsArray : access constant driver_types_h.cudaExternalSemaphoreWaitParams;
      numExtSems : unsigned;
      stream : driver_types_h.cudaStream_t) return driver_types_h.cudaError_t  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/cuda_runtime_api.h:3504
   with Import => True, 
        Convention => C, 
        External_Name => "cudaWaitExternalSemaphoresAsync";

   function cudaDestroyExternalSemaphore (extSem : driver_types_h.cudaExternalSemaphore_t) return driver_types_h.cudaError_t  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/cuda_runtime_api.h:3526
   with Import => True, 
        Convention => C, 
        External_Name => "cudaDestroyExternalSemaphore";

   function cudaLaunchKernel
     (func : System.Address;
      gridDim : vector_types_h.dim3;
      blockDim : vector_types_h.dim3;
      args : System.Address;
      sharedMem : corecrt_h.size_t;
      stream : driver_types_h.cudaStream_t) return driver_types_h.cudaError_t  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/cuda_runtime_api.h:3591
   with Import => True, 
        Convention => C, 
        External_Name => "cudaLaunchKernel";

   function cudaLaunchCooperativeKernel
     (func : System.Address;
      gridDim : vector_types_h.dim3;
      blockDim : vector_types_h.dim3;
      args : System.Address;
      sharedMem : corecrt_h.size_t;
      stream : driver_types_h.cudaStream_t) return driver_types_h.cudaError_t  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/cuda_runtime_api.h:3648
   with Import => True, 
        Convention => C, 
        External_Name => "cudaLaunchCooperativeKernel";

   function cudaLaunchCooperativeKernelMultiDevice
     (launchParamsList : access driver_types_h.cudaLaunchParams;
      numDevices : unsigned;
      flags : unsigned) return driver_types_h.cudaError_t  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/cuda_runtime_api.h:3747
   with Import => True, 
        Convention => C, 
        External_Name => "cudaLaunchCooperativeKernelMultiDevice";

   function cudaFuncSetCacheConfig (func : System.Address; cacheConfig : driver_types_h.cudaFuncCache) return driver_types_h.cudaError_t  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/cuda_runtime_api.h:3796
   with Import => True, 
        Convention => C, 
        External_Name => "cudaFuncSetCacheConfig";

   function cudaFuncSetSharedMemConfig (func : System.Address; config : driver_types_h.cudaSharedMemConfig) return driver_types_h.cudaError_t  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/cuda_runtime_api.h:3851
   with Import => True, 
        Convention => C, 
        External_Name => "cudaFuncSetSharedMemConfig";

   function cudaFuncGetAttributes (attr : access driver_types_h.cudaFuncAttributes; func : System.Address) return driver_types_h.cudaError_t  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/cuda_runtime_api.h:3886
   with Import => True, 
        Convention => C, 
        External_Name => "cudaFuncGetAttributes";

   function cudaFuncSetAttribute
     (func : System.Address;
      attr : driver_types_h.cudaFuncAttribute;
      value : int) return driver_types_h.cudaError_t  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/cuda_runtime_api.h:3925
   with Import => True, 
        Convention => C, 
        External_Name => "cudaFuncSetAttribute";

   function cudaSetDoubleForDevice (d : access double) return driver_types_h.cudaError_t  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/cuda_runtime_api.h:3949
   with Import => True, 
        Convention => C, 
        External_Name => "cudaSetDoubleForDevice";

   function cudaSetDoubleForHost (d : access double) return driver_types_h.cudaError_t  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/cuda_runtime_api.h:3973
   with Import => True, 
        Convention => C, 
        External_Name => "cudaSetDoubleForHost";

   function cudaLaunchHostFunc
     (stream : driver_types_h.cudaStream_t;
      fn : driver_types_h.cudaHostFn_t;
      userData : System.Address) return driver_types_h.cudaError_t  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/cuda_runtime_api.h:4039
   with Import => True, 
        Convention => C, 
        External_Name => "cudaLaunchHostFunc";

   function cudaOccupancyMaxActiveBlocksPerMultiprocessor
     (numBlocks : access int;
      func : System.Address;
      blockSize : int;
      dynamicSMemSize : corecrt_h.size_t) return driver_types_h.cudaError_t  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/cuda_runtime_api.h:4094
   with Import => True, 
        Convention => C, 
        External_Name => "cudaOccupancyMaxActiveBlocksPerMultiprocessor";

   function cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags
     (numBlocks : access int;
      func : System.Address;
      blockSize : int;
      dynamicSMemSize : corecrt_h.size_t;
      flags : unsigned) return driver_types_h.cudaError_t  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/cuda_runtime_api.h:4138
   with Import => True, 
        Convention => C, 
        External_Name => "cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags";

   function cudaMallocManaged
     (devPtr : System.Address;
      size : corecrt_h.size_t;
      flags : unsigned) return driver_types_h.cudaError_t  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/cuda_runtime_api.h:4258
   with Import => True, 
        Convention => C, 
        External_Name => "cudaMallocManaged";

   function cudaMalloc (devPtr : System.Address; size : corecrt_h.size_t) return driver_types_h.cudaError_t  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/cuda_runtime_api.h:4289
   with Import => True, 
        Convention => C, 
        External_Name => "cudaMalloc";

   function cudaMallocHost (ptr : System.Address; size : corecrt_h.size_t) return driver_types_h.cudaError_t  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/cuda_runtime_api.h:4322
   with Import => True, 
        Convention => C, 
        External_Name => "cudaMallocHost";

   function cudaMallocPitch
     (devPtr : System.Address;
      pitch : access corecrt_h.size_t;
      width : corecrt_h.size_t;
      height : corecrt_h.size_t) return driver_types_h.cudaError_t  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/cuda_runtime_api.h:4365
   with Import => True, 
        Convention => C, 
        External_Name => "cudaMallocPitch";

   function cudaMallocArray
     (c_array : System.Address;
      desc : access constant driver_types_h.cudaChannelFormatDesc;
      width : corecrt_h.size_t;
      height : corecrt_h.size_t;
      flags : unsigned) return driver_types_h.cudaError_t  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/cuda_runtime_api.h:4411
   with Import => True, 
        Convention => C, 
        External_Name => "cudaMallocArray";

   function cudaFree (devPtr : System.Address) return driver_types_h.cudaError_t  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/cuda_runtime_api.h:4440
   with Import => True, 
        Convention => C, 
        External_Name => "cudaFree";

   function cudaFreeHost (ptr : System.Address) return driver_types_h.cudaError_t  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/cuda_runtime_api.h:4463
   with Import => True, 
        Convention => C, 
        External_Name => "cudaFreeHost";

   function cudaFreeArray (c_array : driver_types_h.cudaArray_t) return driver_types_h.cudaError_t  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/cuda_runtime_api.h:4486
   with Import => True, 
        Convention => C, 
        External_Name => "cudaFreeArray";

   function cudaFreeMipmappedArray (mipmappedArray : driver_types_h.cudaMipmappedArray_t) return driver_types_h.cudaError_t  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/cuda_runtime_api.h:4509
   with Import => True, 
        Convention => C, 
        External_Name => "cudaFreeMipmappedArray";

   function cudaHostAlloc
     (pHost : System.Address;
      size : corecrt_h.size_t;
      flags : unsigned) return driver_types_h.cudaError_t  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/cuda_runtime_api.h:4575
   with Import => True, 
        Convention => C, 
        External_Name => "cudaHostAlloc";

   function cudaHostRegister
     (ptr : System.Address;
      size : corecrt_h.size_t;
      flags : unsigned) return driver_types_h.cudaError_t  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/cuda_runtime_api.h:4659
   with Import => True, 
        Convention => C, 
        External_Name => "cudaHostRegister";

   function cudaHostUnregister (ptr : System.Address) return driver_types_h.cudaError_t  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/cuda_runtime_api.h:4682
   with Import => True, 
        Convention => C, 
        External_Name => "cudaHostUnregister";

   function cudaHostGetDevicePointer
     (pDevice : System.Address;
      pHost : System.Address;
      flags : unsigned) return driver_types_h.cudaError_t  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/cuda_runtime_api.h:4727
   with Import => True, 
        Convention => C, 
        External_Name => "cudaHostGetDevicePointer";

   function cudaHostGetFlags (pFlags : access unsigned; pHost : System.Address) return driver_types_h.cudaError_t  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/cuda_runtime_api.h:4749
   with Import => True, 
        Convention => C, 
        External_Name => "cudaHostGetFlags";

   function cudaMalloc3D (pitchedDevPtr : access driver_types_h.cudaPitchedPtr; extent : driver_types_h.cudaExtent) return driver_types_h.cudaError_t  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/cuda_runtime_api.h:4788
   with Import => True, 
        Convention => C, 
        External_Name => "cudaMalloc3D";

   function cudaMalloc3DArray
     (c_array : System.Address;
      desc : access constant driver_types_h.cudaChannelFormatDesc;
      extent : driver_types_h.cudaExtent;
      flags : unsigned) return driver_types_h.cudaError_t  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/cuda_runtime_api.h:4927
   with Import => True, 
        Convention => C, 
        External_Name => "cudaMalloc3DArray";

   function cudaMallocMipmappedArray
     (mipmappedArray : System.Address;
      desc : access constant driver_types_h.cudaChannelFormatDesc;
      extent : driver_types_h.cudaExtent;
      numLevels : unsigned;
      flags : unsigned) return driver_types_h.cudaError_t  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/cuda_runtime_api.h:5066
   with Import => True, 
        Convention => C, 
        External_Name => "cudaMallocMipmappedArray";

   function cudaGetMipmappedArrayLevel
     (levelArray : System.Address;
      mipmappedArray : driver_types_h.cudaMipmappedArray_const_t;
      level : unsigned) return driver_types_h.cudaError_t  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/cuda_runtime_api.h:5095
   with Import => True, 
        Convention => C, 
        External_Name => "cudaGetMipmappedArrayLevel";

   function cudaMemcpy3D (p : access constant driver_types_h.cudaMemcpy3DParms) return driver_types_h.cudaError_t  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/cuda_runtime_api.h:5200
   with Import => True, 
        Convention => C, 
        External_Name => "cudaMemcpy3D";

   function cudaMemcpy3DPeer (p : access constant driver_types_h.cudaMemcpy3DPeerParms) return driver_types_h.cudaError_t  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/cuda_runtime_api.h:5231
   with Import => True, 
        Convention => C, 
        External_Name => "cudaMemcpy3DPeer";

   function cudaMemcpy3DAsync (p : access constant driver_types_h.cudaMemcpy3DParms; stream : driver_types_h.cudaStream_t) return driver_types_h.cudaError_t  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/cuda_runtime_api.h:5349
   with Import => True, 
        Convention => C, 
        External_Name => "cudaMemcpy3DAsync";

   function cudaMemcpy3DPeerAsync (p : access constant driver_types_h.cudaMemcpy3DPeerParms; stream : driver_types_h.cudaStream_t) return driver_types_h.cudaError_t  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/cuda_runtime_api.h:5375
   with Import => True, 
        Convention => C, 
        External_Name => "cudaMemcpy3DPeerAsync";

   function cudaMemGetInfo (free : access corecrt_h.size_t; total : access corecrt_h.size_t) return driver_types_h.cudaError_t  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/cuda_runtime_api.h:5397
   with Import => True, 
        Convention => C, 
        External_Name => "cudaMemGetInfo";

   function cudaArrayGetInfo
     (desc : access driver_types_h.cudaChannelFormatDesc;
      extent : access driver_types_h.cudaExtent;
      flags : access unsigned;
      c_array : driver_types_h.cudaArray_t) return driver_types_h.cudaError_t  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/cuda_runtime_api.h:5423
   with Import => True, 
        Convention => C, 
        External_Name => "cudaArrayGetInfo";

   function cudaMemcpy
     (dst : System.Address;
      src : System.Address;
      count : corecrt_h.size_t;
      kind : driver_types_h.cudaMemcpyKind) return driver_types_h.cudaError_t  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/cuda_runtime_api.h:5466
   with Import => True, 
        Convention => C, 
        External_Name => "cudaMemcpy";

   function cudaMemcpyPeer
     (dst : System.Address;
      dstDevice : int;
      src : System.Address;
      srcDevice : int;
      count : corecrt_h.size_t) return driver_types_h.cudaError_t  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/cuda_runtime_api.h:5501
   with Import => True, 
        Convention => C, 
        External_Name => "cudaMemcpyPeer";

   function cudaMemcpy2D
     (dst : System.Address;
      dpitch : corecrt_h.size_t;
      src : System.Address;
      spitch : corecrt_h.size_t;
      width : corecrt_h.size_t;
      height : corecrt_h.size_t;
      kind : driver_types_h.cudaMemcpyKind) return driver_types_h.cudaError_t  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/cuda_runtime_api.h:5549
   with Import => True, 
        Convention => C, 
        External_Name => "cudaMemcpy2D";

   function cudaMemcpy2DToArray
     (dst : driver_types_h.cudaArray_t;
      wOffset : corecrt_h.size_t;
      hOffset : corecrt_h.size_t;
      src : System.Address;
      spitch : corecrt_h.size_t;
      width : corecrt_h.size_t;
      height : corecrt_h.size_t;
      kind : driver_types_h.cudaMemcpyKind) return driver_types_h.cudaError_t  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/cuda_runtime_api.h:5598
   with Import => True, 
        Convention => C, 
        External_Name => "cudaMemcpy2DToArray";

   function cudaMemcpy2DFromArray
     (dst : System.Address;
      dpitch : corecrt_h.size_t;
      src : driver_types_h.cudaArray_const_t;
      wOffset : corecrt_h.size_t;
      hOffset : corecrt_h.size_t;
      width : corecrt_h.size_t;
      height : corecrt_h.size_t;
      kind : driver_types_h.cudaMemcpyKind) return driver_types_h.cudaError_t  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/cuda_runtime_api.h:5647
   with Import => True, 
        Convention => C, 
        External_Name => "cudaMemcpy2DFromArray";

   function cudaMemcpy2DArrayToArray
     (dst : driver_types_h.cudaArray_t;
      wOffsetDst : corecrt_h.size_t;
      hOffsetDst : corecrt_h.size_t;
      src : driver_types_h.cudaArray_const_t;
      wOffsetSrc : corecrt_h.size_t;
      hOffsetSrc : corecrt_h.size_t;
      width : corecrt_h.size_t;
      height : corecrt_h.size_t;
      kind : driver_types_h.cudaMemcpyKind) return driver_types_h.cudaError_t  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/cuda_runtime_api.h:5694
   with Import => True, 
        Convention => C, 
        External_Name => "cudaMemcpy2DArrayToArray";

   function cudaMemcpyToSymbol
     (symbol : System.Address;
      src : System.Address;
      count : corecrt_h.size_t;
      offset : corecrt_h.size_t;
      kind : driver_types_h.cudaMemcpyKind) return driver_types_h.cudaError_t  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/cuda_runtime_api.h:5737
   with Import => True, 
        Convention => C, 
        External_Name => "cudaMemcpyToSymbol";

   function cudaMemcpyFromSymbol
     (dst : System.Address;
      symbol : System.Address;
      count : corecrt_h.size_t;
      offset : corecrt_h.size_t;
      kind : driver_types_h.cudaMemcpyKind) return driver_types_h.cudaError_t  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/cuda_runtime_api.h:5780
   with Import => True, 
        Convention => C, 
        External_Name => "cudaMemcpyFromSymbol";

   function cudaMemcpyAsync
     (dst : System.Address;
      src : System.Address;
      count : corecrt_h.size_t;
      kind : driver_types_h.cudaMemcpyKind;
      stream : driver_types_h.cudaStream_t) return driver_types_h.cudaError_t  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/cuda_runtime_api.h:5836
   with Import => True, 
        Convention => C, 
        External_Name => "cudaMemcpyAsync";

   function cudaMemcpyPeerAsync
     (dst : System.Address;
      dstDevice : int;
      src : System.Address;
      srcDevice : int;
      count : corecrt_h.size_t;
      stream : driver_types_h.cudaStream_t) return driver_types_h.cudaError_t  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/cuda_runtime_api.h:5871
   with Import => True, 
        Convention => C, 
        External_Name => "cudaMemcpyPeerAsync";

   function cudaMemcpy2DAsync
     (dst : System.Address;
      dpitch : corecrt_h.size_t;
      src : System.Address;
      spitch : corecrt_h.size_t;
      width : corecrt_h.size_t;
      height : corecrt_h.size_t;
      kind : driver_types_h.cudaMemcpyKind;
      stream : driver_types_h.cudaStream_t) return driver_types_h.cudaError_t  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/cuda_runtime_api.h:5933
   with Import => True, 
        Convention => C, 
        External_Name => "cudaMemcpy2DAsync";

   function cudaMemcpy2DToArrayAsync
     (dst : driver_types_h.cudaArray_t;
      wOffset : corecrt_h.size_t;
      hOffset : corecrt_h.size_t;
      src : System.Address;
      spitch : corecrt_h.size_t;
      width : corecrt_h.size_t;
      height : corecrt_h.size_t;
      kind : driver_types_h.cudaMemcpyKind;
      stream : driver_types_h.cudaStream_t) return driver_types_h.cudaError_t  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/cuda_runtime_api.h:5990
   with Import => True, 
        Convention => C, 
        External_Name => "cudaMemcpy2DToArrayAsync";

   function cudaMemcpy2DFromArrayAsync
     (dst : System.Address;
      dpitch : corecrt_h.size_t;
      src : driver_types_h.cudaArray_const_t;
      wOffset : corecrt_h.size_t;
      hOffset : corecrt_h.size_t;
      width : corecrt_h.size_t;
      height : corecrt_h.size_t;
      kind : driver_types_h.cudaMemcpyKind;
      stream : driver_types_h.cudaStream_t) return driver_types_h.cudaError_t  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/cuda_runtime_api.h:6046
   with Import => True, 
        Convention => C, 
        External_Name => "cudaMemcpy2DFromArrayAsync";

   function cudaMemcpyToSymbolAsync
     (symbol : System.Address;
      src : System.Address;
      count : corecrt_h.size_t;
      offset : corecrt_h.size_t;
      kind : driver_types_h.cudaMemcpyKind;
      stream : driver_types_h.cudaStream_t) return driver_types_h.cudaError_t  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/cuda_runtime_api.h:6097
   with Import => True, 
        Convention => C, 
        External_Name => "cudaMemcpyToSymbolAsync";

   function cudaMemcpyFromSymbolAsync
     (dst : System.Address;
      symbol : System.Address;
      count : corecrt_h.size_t;
      offset : corecrt_h.size_t;
      kind : driver_types_h.cudaMemcpyKind;
      stream : driver_types_h.cudaStream_t) return driver_types_h.cudaError_t  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/cuda_runtime_api.h:6148
   with Import => True, 
        Convention => C, 
        External_Name => "cudaMemcpyFromSymbolAsync";

   function cudaMemset
     (devPtr : System.Address;
      value : int;
      count : corecrt_h.size_t) return driver_types_h.cudaError_t  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/cuda_runtime_api.h:6177
   with Import => True, 
        Convention => C, 
        External_Name => "cudaMemset";

   function cudaMemset2D
     (devPtr : System.Address;
      pitch : corecrt_h.size_t;
      value : int;
      width : corecrt_h.size_t;
      height : corecrt_h.size_t) return driver_types_h.cudaError_t  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/cuda_runtime_api.h:6211
   with Import => True, 
        Convention => C, 
        External_Name => "cudaMemset2D";

   function cudaMemset3D
     (pitchedDevPtr : driver_types_h.cudaPitchedPtr;
      value : int;
      extent : driver_types_h.cudaExtent) return driver_types_h.cudaError_t  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/cuda_runtime_api.h:6255
   with Import => True, 
        Convention => C, 
        External_Name => "cudaMemset3D";

   function cudaMemsetAsync
     (devPtr : System.Address;
      value : int;
      count : corecrt_h.size_t;
      stream : driver_types_h.cudaStream_t) return driver_types_h.cudaError_t  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/cuda_runtime_api.h:6291
   with Import => True, 
        Convention => C, 
        External_Name => "cudaMemsetAsync";

   function cudaMemset2DAsync
     (devPtr : System.Address;
      pitch : corecrt_h.size_t;
      value : int;
      width : corecrt_h.size_t;
      height : corecrt_h.size_t;
      stream : driver_types_h.cudaStream_t) return driver_types_h.cudaError_t  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/cuda_runtime_api.h:6332
   with Import => True, 
        Convention => C, 
        External_Name => "cudaMemset2DAsync";

   function cudaMemset3DAsync
     (pitchedDevPtr : driver_types_h.cudaPitchedPtr;
      value : int;
      extent : driver_types_h.cudaExtent;
      stream : driver_types_h.cudaStream_t) return driver_types_h.cudaError_t  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/cuda_runtime_api.h:6383
   with Import => True, 
        Convention => C, 
        External_Name => "cudaMemset3DAsync";

   function cudaGetSymbolAddress (devPtr : System.Address; symbol : System.Address) return driver_types_h.cudaError_t  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/cuda_runtime_api.h:6411
   with Import => True, 
        Convention => C, 
        External_Name => "cudaGetSymbolAddress";

   function cudaGetSymbolSize (size : access corecrt_h.size_t; symbol : System.Address) return driver_types_h.cudaError_t  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/cuda_runtime_api.h:6438
   with Import => True, 
        Convention => C, 
        External_Name => "cudaGetSymbolSize";

   function cudaMemPrefetchAsync
     (devPtr : System.Address;
      count : corecrt_h.size_t;
      dstDevice : int;
      stream : driver_types_h.cudaStream_t) return driver_types_h.cudaError_t  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/cuda_runtime_api.h:6508
   with Import => True, 
        Convention => C, 
        External_Name => "cudaMemPrefetchAsync";

   function cudaMemAdvise
     (devPtr : System.Address;
      count : corecrt_h.size_t;
      advice : driver_types_h.cudaMemoryAdvise;
      device : int) return driver_types_h.cudaError_t  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/cuda_runtime_api.h:6624
   with Import => True, 
        Convention => C, 
        External_Name => "cudaMemAdvise";

   function cudaMemRangeGetAttribute
     (data : System.Address;
      dataSize : corecrt_h.size_t;
      attribute : driver_types_h.cudaMemRangeAttribute;
      devPtr : System.Address;
      count : corecrt_h.size_t) return driver_types_h.cudaError_t  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/cuda_runtime_api.h:6683
   with Import => True, 
        Convention => C, 
        External_Name => "cudaMemRangeGetAttribute";

   function cudaMemRangeGetAttributes
     (data : System.Address;
      dataSizes : access corecrt_h.size_t;
      attributes : access driver_types_h.cudaMemRangeAttribute;
      numAttributes : corecrt_h.size_t;
      devPtr : System.Address;
      count : corecrt_h.size_t) return driver_types_h.cudaError_t  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/cuda_runtime_api.h:6722
   with Import => True, 
        Convention => C, 
        External_Name => "cudaMemRangeGetAttributes";

   function cudaMemcpyToArray
     (dst : driver_types_h.cudaArray_t;
      wOffset : corecrt_h.size_t;
      hOffset : corecrt_h.size_t;
      src : System.Address;
      count : corecrt_h.size_t;
      kind : driver_types_h.cudaMemcpyKind) return driver_types_h.cudaError_t  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/cuda_runtime_api.h:6782
   with Import => True, 
        Convention => C, 
        External_Name => "cudaMemcpyToArray";

   function cudaMemcpyFromArray
     (dst : System.Address;
      src : driver_types_h.cudaArray_const_t;
      wOffset : corecrt_h.size_t;
      hOffset : corecrt_h.size_t;
      count : corecrt_h.size_t;
      kind : driver_types_h.cudaMemcpyKind) return driver_types_h.cudaError_t  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/cuda_runtime_api.h:6824
   with Import => True, 
        Convention => C, 
        External_Name => "cudaMemcpyFromArray";

   function cudaMemcpyArrayToArray
     (dst : driver_types_h.cudaArray_t;
      wOffsetDst : corecrt_h.size_t;
      hOffsetDst : corecrt_h.size_t;
      src : driver_types_h.cudaArray_const_t;
      wOffsetSrc : corecrt_h.size_t;
      hOffsetSrc : corecrt_h.size_t;
      count : corecrt_h.size_t;
      kind : driver_types_h.cudaMemcpyKind) return driver_types_h.cudaError_t  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/cuda_runtime_api.h:6867
   with Import => True, 
        Convention => C, 
        External_Name => "cudaMemcpyArrayToArray";

   function cudaMemcpyToArrayAsync
     (dst : driver_types_h.cudaArray_t;
      wOffset : corecrt_h.size_t;
      hOffset : corecrt_h.size_t;
      src : System.Address;
      count : corecrt_h.size_t;
      kind : driver_types_h.cudaMemcpyKind;
      stream : driver_types_h.cudaStream_t) return driver_types_h.cudaError_t  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/cuda_runtime_api.h:6918
   with Import => True, 
        Convention => C, 
        External_Name => "cudaMemcpyToArrayAsync";

   function cudaMemcpyFromArrayAsync
     (dst : System.Address;
      src : driver_types_h.cudaArray_const_t;
      wOffset : corecrt_h.size_t;
      hOffset : corecrt_h.size_t;
      count : corecrt_h.size_t;
      kind : driver_types_h.cudaMemcpyKind;
      stream : driver_types_h.cudaStream_t) return driver_types_h.cudaError_t  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/cuda_runtime_api.h:6968
   with Import => True, 
        Convention => C, 
        External_Name => "cudaMemcpyFromArrayAsync";

   function cudaPointerGetAttributes (attributes : access driver_types_h.cudaPointerAttributes; ptr : System.Address) return driver_types_h.cudaError_t  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/cuda_runtime_api.h:7134
   with Import => True, 
        Convention => C, 
        External_Name => "cudaPointerGetAttributes";

   function cudaDeviceCanAccessPeer
     (canAccessPeer : access int;
      device : int;
      peerDevice : int) return driver_types_h.cudaError_t  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/cuda_runtime_api.h:7175
   with Import => True, 
        Convention => C, 
        External_Name => "cudaDeviceCanAccessPeer";

   function cudaDeviceEnablePeerAccess (peerDevice : int; flags : unsigned) return driver_types_h.cudaError_t  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/cuda_runtime_api.h:7217
   with Import => True, 
        Convention => C, 
        External_Name => "cudaDeviceEnablePeerAccess";

   function cudaDeviceDisablePeerAccess (peerDevice : int) return driver_types_h.cudaError_t  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/cuda_runtime_api.h:7239
   with Import => True, 
        Convention => C, 
        External_Name => "cudaDeviceDisablePeerAccess";

   function cudaGraphicsUnregisterResource (resource : driver_types_h.cudaGraphicsResource_t) return driver_types_h.cudaError_t  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/cuda_runtime_api.h:7302
   with Import => True, 
        Convention => C, 
        External_Name => "cudaGraphicsUnregisterResource";

   function cudaGraphicsResourceSetMapFlags (resource : driver_types_h.cudaGraphicsResource_t; flags : unsigned) return driver_types_h.cudaError_t  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/cuda_runtime_api.h:7337
   with Import => True, 
        Convention => C, 
        External_Name => "cudaGraphicsResourceSetMapFlags";

   function cudaGraphicsMapResources
     (count : int;
      resources : System.Address;
      stream : driver_types_h.cudaStream_t) return driver_types_h.cudaError_t  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/cuda_runtime_api.h:7376
   with Import => True, 
        Convention => C, 
        External_Name => "cudaGraphicsMapResources";

   function cudaGraphicsUnmapResources
     (count : int;
      resources : System.Address;
      stream : driver_types_h.cudaStream_t) return driver_types_h.cudaError_t  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/cuda_runtime_api.h:7411
   with Import => True, 
        Convention => C, 
        External_Name => "cudaGraphicsUnmapResources";

   function cudaGraphicsResourceGetMappedPointer
     (devPtr : System.Address;
      size : access corecrt_h.size_t;
      resource : driver_types_h.cudaGraphicsResource_t) return driver_types_h.cudaError_t  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/cuda_runtime_api.h:7443
   with Import => True, 
        Convention => C, 
        External_Name => "cudaGraphicsResourceGetMappedPointer";

   function cudaGraphicsSubResourceGetMappedArray
     (c_array : System.Address;
      resource : driver_types_h.cudaGraphicsResource_t;
      arrayIndex : unsigned;
      mipLevel : unsigned) return driver_types_h.cudaError_t  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/cuda_runtime_api.h:7481
   with Import => True, 
        Convention => C, 
        External_Name => "cudaGraphicsSubResourceGetMappedArray";

   function cudaGraphicsResourceGetMappedMipmappedArray (mipmappedArray : System.Address; resource : driver_types_h.cudaGraphicsResource_t) return driver_types_h.cudaError_t  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/cuda_runtime_api.h:7510
   with Import => True, 
        Convention => C, 
        External_Name => "cudaGraphicsResourceGetMappedMipmappedArray";

   function cudaBindTexture
     (offset : access corecrt_h.size_t;
      texref : access constant texture_types_h.textureReference;
      devPtr : System.Address;
      desc : access constant driver_types_h.cudaChannelFormatDesc;
      size : corecrt_h.size_t) return driver_types_h.cudaError_t  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/cuda_runtime_api.h:7581
   with Import => True, 
        Convention => C, 
        External_Name => "cudaBindTexture";

   function cudaBindTexture2D
     (offset : access corecrt_h.size_t;
      texref : access constant texture_types_h.textureReference;
      devPtr : System.Address;
      desc : access constant driver_types_h.cudaChannelFormatDesc;
      width : corecrt_h.size_t;
      height : corecrt_h.size_t;
      pitch : corecrt_h.size_t) return driver_types_h.cudaError_t  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/cuda_runtime_api.h:7640
   with Import => True, 
        Convention => C, 
        External_Name => "cudaBindTexture2D";

   function cudaBindTextureToArray
     (texref : access constant texture_types_h.textureReference;
      c_array : driver_types_h.cudaArray_const_t;
      desc : access constant driver_types_h.cudaChannelFormatDesc) return driver_types_h.cudaError_t  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/cuda_runtime_api.h:7678
   with Import => True, 
        Convention => C, 
        External_Name => "cudaBindTextureToArray";

   function cudaBindTextureToMipmappedArray
     (texref : access constant texture_types_h.textureReference;
      mipmappedArray : driver_types_h.cudaMipmappedArray_const_t;
      desc : access constant driver_types_h.cudaChannelFormatDesc) return driver_types_h.cudaError_t  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/cuda_runtime_api.h:7718
   with Import => True, 
        Convention => C, 
        External_Name => "cudaBindTextureToMipmappedArray";

   function cudaUnbindTexture (texref : access constant texture_types_h.textureReference) return driver_types_h.cudaError_t  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/cuda_runtime_api.h:7744
   with Import => True, 
        Convention => C, 
        External_Name => "cudaUnbindTexture";

   function cudaGetTextureAlignmentOffset (offset : access corecrt_h.size_t; texref : access constant texture_types_h.textureReference) return driver_types_h.cudaError_t  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/cuda_runtime_api.h:7773
   with Import => True, 
        Convention => C, 
        External_Name => "cudaGetTextureAlignmentOffset";

   function cudaGetTextureReference (texref : System.Address; symbol : System.Address) return driver_types_h.cudaError_t  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/cuda_runtime_api.h:7803
   with Import => True, 
        Convention => C, 
        External_Name => "cudaGetTextureReference";

   function cudaBindSurfaceToArray
     (surfref : access constant surface_types_h.surfaceReference;
      c_array : driver_types_h.cudaArray_const_t;
      desc : access constant driver_types_h.cudaChannelFormatDesc) return driver_types_h.cudaError_t  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/cuda_runtime_api.h:7848
   with Import => True, 
        Convention => C, 
        External_Name => "cudaBindSurfaceToArray";

   function cudaGetSurfaceReference (surfref : System.Address; symbol : System.Address) return driver_types_h.cudaError_t  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/cuda_runtime_api.h:7873
   with Import => True, 
        Convention => C, 
        External_Name => "cudaGetSurfaceReference";

   function cudaGetChannelDesc (desc : access driver_types_h.cudaChannelFormatDesc; c_array : driver_types_h.cudaArray_const_t) return driver_types_h.cudaError_t  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/cuda_runtime_api.h:7908
   with Import => True, 
        Convention => C, 
        External_Name => "cudaGetChannelDesc";

   function cudaCreateChannelDesc
     (x : int;
      y : int;
      z : int;
      w : int;
      f : driver_types_h.cudaChannelFormatKind) return driver_types_h.cudaChannelFormatDesc  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/cuda_runtime_api.h:7938
   with Import => True, 
        Convention => C, 
        External_Name => "cudaCreateChannelDesc";

   function cudaCreateTextureObject
     (pTexObject : access texture_types_h.cudaTextureObject_t;
      pResDesc : access constant driver_types_h.cudaResourceDesc;
      pTexDesc : access constant texture_types_h.cudaTextureDesc;
      pResViewDesc : access constant driver_types_h.cudaResourceViewDesc) return driver_types_h.cudaError_t  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/cuda_runtime_api.h:8153
   with Import => True, 
        Convention => C, 
        External_Name => "cudaCreateTextureObject";

   function cudaDestroyTextureObject (texObject : texture_types_h.cudaTextureObject_t) return driver_types_h.cudaError_t  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/cuda_runtime_api.h:8172
   with Import => True, 
        Convention => C, 
        External_Name => "cudaDestroyTextureObject";

   function cudaGetTextureObjectResourceDesc (pResDesc : access driver_types_h.cudaResourceDesc; texObject : texture_types_h.cudaTextureObject_t) return driver_types_h.cudaError_t  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/cuda_runtime_api.h:8192
   with Import => True, 
        Convention => C, 
        External_Name => "cudaGetTextureObjectResourceDesc";

   function cudaGetTextureObjectTextureDesc (pTexDesc : access texture_types_h.cudaTextureDesc; texObject : texture_types_h.cudaTextureObject_t) return driver_types_h.cudaError_t  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/cuda_runtime_api.h:8212
   with Import => True, 
        Convention => C, 
        External_Name => "cudaGetTextureObjectTextureDesc";

   function cudaGetTextureObjectResourceViewDesc (pResViewDesc : access driver_types_h.cudaResourceViewDesc; texObject : texture_types_h.cudaTextureObject_t) return driver_types_h.cudaError_t  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/cuda_runtime_api.h:8233
   with Import => True, 
        Convention => C, 
        External_Name => "cudaGetTextureObjectResourceViewDesc";

   function cudaCreateSurfaceObject (pSurfObject : access surface_types_h.cudaSurfaceObject_t; pResDesc : access constant driver_types_h.cudaResourceDesc) return driver_types_h.cudaError_t  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/cuda_runtime_api.h:8278
   with Import => True, 
        Convention => C, 
        External_Name => "cudaCreateSurfaceObject";

   function cudaDestroySurfaceObject (surfObject : surface_types_h.cudaSurfaceObject_t) return driver_types_h.cudaError_t  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/cuda_runtime_api.h:8297
   with Import => True, 
        Convention => C, 
        External_Name => "cudaDestroySurfaceObject";

   function cudaGetSurfaceObjectResourceDesc (pResDesc : access driver_types_h.cudaResourceDesc; surfObject : surface_types_h.cudaSurfaceObject_t) return driver_types_h.cudaError_t  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/cuda_runtime_api.h:8316
   with Import => True, 
        Convention => C, 
        External_Name => "cudaGetSurfaceObjectResourceDesc";

   function cudaDriverGetVersion (driverVersion : access int) return driver_types_h.cudaError_t  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/cuda_runtime_api.h:8350
   with Import => True, 
        Convention => C, 
        External_Name => "cudaDriverGetVersion";

   function cudaRuntimeGetVersion (runtimeVersion : access int) return driver_types_h.cudaError_t  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/cuda_runtime_api.h:8375
   with Import => True, 
        Convention => C, 
        External_Name => "cudaRuntimeGetVersion";

   function cudaGraphCreate (pGraph : System.Address; flags : unsigned) return driver_types_h.cudaError_t  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/cuda_runtime_api.h:8422
   with Import => True, 
        Convention => C, 
        External_Name => "cudaGraphCreate";

   function cudaGraphAddKernelNode
     (pGraphNode : System.Address;
      graph : driver_types_h.cudaGraph_t;
      pDependencies : System.Address;
      numDependencies : corecrt_h.size_t;
      pNodeParams : access constant driver_types_h.cudaKernelNodeParams) return driver_types_h.cudaError_t  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/cuda_runtime_api.h:8519
   with Import => True, 
        Convention => C, 
        External_Name => "cudaGraphAddKernelNode";

   function cudaGraphKernelNodeGetParams (node : driver_types_h.cudaGraphNode_t; pNodeParams : access driver_types_h.cudaKernelNodeParams) return driver_types_h.cudaError_t  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/cuda_runtime_api.h:8552
   with Import => True, 
        Convention => C, 
        External_Name => "cudaGraphKernelNodeGetParams";

   function cudaGraphKernelNodeSetParams (node : driver_types_h.cudaGraphNode_t; pNodeParams : access constant driver_types_h.cudaKernelNodeParams) return driver_types_h.cudaError_t  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/cuda_runtime_api.h:8577
   with Import => True, 
        Convention => C, 
        External_Name => "cudaGraphKernelNodeSetParams";

   function cudaGraphAddMemcpyNode
     (pGraphNode : System.Address;
      graph : driver_types_h.cudaGraph_t;
      pDependencies : System.Address;
      numDependencies : corecrt_h.size_t;
      pCopyParams : access constant driver_types_h.cudaMemcpy3DParms) return driver_types_h.cudaError_t  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/cuda_runtime_api.h:8621
   with Import => True, 
        Convention => C, 
        External_Name => "cudaGraphAddMemcpyNode";

   function cudaGraphMemcpyNodeGetParams (node : driver_types_h.cudaGraphNode_t; pNodeParams : access driver_types_h.cudaMemcpy3DParms) return driver_types_h.cudaError_t  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/cuda_runtime_api.h:8644
   with Import => True, 
        Convention => C, 
        External_Name => "cudaGraphMemcpyNodeGetParams";

   function cudaGraphMemcpyNodeSetParams (node : driver_types_h.cudaGraphNode_t; pNodeParams : access constant driver_types_h.cudaMemcpy3DParms) return driver_types_h.cudaError_t  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/cuda_runtime_api.h:8667
   with Import => True, 
        Convention => C, 
        External_Name => "cudaGraphMemcpyNodeSetParams";

   function cudaGraphAddMemsetNode
     (pGraphNode : System.Address;
      graph : driver_types_h.cudaGraph_t;
      pDependencies : System.Address;
      numDependencies : corecrt_h.size_t;
      pMemsetParams : access constant driver_types_h.cudaMemsetParams) return driver_types_h.cudaError_t  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/cuda_runtime_api.h:8709
   with Import => True, 
        Convention => C, 
        External_Name => "cudaGraphAddMemsetNode";

   function cudaGraphMemsetNodeGetParams (node : driver_types_h.cudaGraphNode_t; pNodeParams : access driver_types_h.cudaMemsetParams) return driver_types_h.cudaError_t  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/cuda_runtime_api.h:8732
   with Import => True, 
        Convention => C, 
        External_Name => "cudaGraphMemsetNodeGetParams";

   function cudaGraphMemsetNodeSetParams (node : driver_types_h.cudaGraphNode_t; pNodeParams : access constant driver_types_h.cudaMemsetParams) return driver_types_h.cudaError_t  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/cuda_runtime_api.h:8755
   with Import => True, 
        Convention => C, 
        External_Name => "cudaGraphMemsetNodeSetParams";

   function cudaGraphAddHostNode
     (pGraphNode : System.Address;
      graph : driver_types_h.cudaGraph_t;
      pDependencies : System.Address;
      numDependencies : corecrt_h.size_t;
      pNodeParams : access constant driver_types_h.cudaHostNodeParams) return driver_types_h.cudaError_t  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/cuda_runtime_api.h:8796
   with Import => True, 
        Convention => C, 
        External_Name => "cudaGraphAddHostNode";

   function cudaGraphHostNodeGetParams (node : driver_types_h.cudaGraphNode_t; pNodeParams : access driver_types_h.cudaHostNodeParams) return driver_types_h.cudaError_t  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/cuda_runtime_api.h:8819
   with Import => True, 
        Convention => C, 
        External_Name => "cudaGraphHostNodeGetParams";

   function cudaGraphHostNodeSetParams (node : driver_types_h.cudaGraphNode_t; pNodeParams : access constant driver_types_h.cudaHostNodeParams) return driver_types_h.cudaError_t  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/cuda_runtime_api.h:8842
   with Import => True, 
        Convention => C, 
        External_Name => "cudaGraphHostNodeSetParams";

   function cudaGraphAddChildGraphNode
     (pGraphNode : System.Address;
      graph : driver_types_h.cudaGraph_t;
      pDependencies : System.Address;
      numDependencies : corecrt_h.size_t;
      childGraph : driver_types_h.cudaGraph_t) return driver_types_h.cudaError_t  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/cuda_runtime_api.h:8880
   with Import => True, 
        Convention => C, 
        External_Name => "cudaGraphAddChildGraphNode";

   function cudaGraphChildGraphNodeGetGraph (node : driver_types_h.cudaGraphNode_t; pGraph : System.Address) return driver_types_h.cudaError_t  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/cuda_runtime_api.h:8904
   with Import => True, 
        Convention => C, 
        External_Name => "cudaGraphChildGraphNodeGetGraph";

   function cudaGraphAddEmptyNode
     (pGraphNode : System.Address;
      graph : driver_types_h.cudaGraph_t;
      pDependencies : System.Address;
      numDependencies : corecrt_h.size_t) return driver_types_h.cudaError_t  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/cuda_runtime_api.h:8941
   with Import => True, 
        Convention => C, 
        External_Name => "cudaGraphAddEmptyNode";

   function cudaGraphClone (pGraphClone : System.Address; originalGraph : driver_types_h.cudaGraph_t) return driver_types_h.cudaError_t  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/cuda_runtime_api.h:8968
   with Import => True, 
        Convention => C, 
        External_Name => "cudaGraphClone";

   function cudaGraphNodeFindInClone
     (pNode : System.Address;
      originalNode : driver_types_h.cudaGraphNode_t;
      clonedGraph : driver_types_h.cudaGraph_t) return driver_types_h.cudaError_t  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/cuda_runtime_api.h:8996
   with Import => True, 
        Convention => C, 
        External_Name => "cudaGraphNodeFindInClone";

   function cudaGraphNodeGetType (node : driver_types_h.cudaGraphNode_t; pType : access driver_types_h.cudaGraphNodeType) return driver_types_h.cudaError_t  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/cuda_runtime_api.h:9027
   with Import => True, 
        Convention => C, 
        External_Name => "cudaGraphNodeGetType";

   function cudaGraphGetNodes
     (graph : driver_types_h.cudaGraph_t;
      nodes : System.Address;
      numNodes : access corecrt_h.size_t) return driver_types_h.cudaError_t  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/cuda_runtime_api.h:9058
   with Import => True, 
        Convention => C, 
        External_Name => "cudaGraphGetNodes";

   function cudaGraphGetRootNodes
     (graph : driver_types_h.cudaGraph_t;
      pRootNodes : System.Address;
      pNumRootNodes : access corecrt_h.size_t) return driver_types_h.cudaError_t  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/cuda_runtime_api.h:9089
   with Import => True, 
        Convention => C, 
        External_Name => "cudaGraphGetRootNodes";

   function cudaGraphGetEdges
     (graph : driver_types_h.cudaGraph_t;
      from : System.Address;
      to : System.Address;
      numEdges : access corecrt_h.size_t) return driver_types_h.cudaError_t  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/cuda_runtime_api.h:9123
   with Import => True, 
        Convention => C, 
        External_Name => "cudaGraphGetEdges";

   function cudaGraphNodeGetDependencies
     (node : driver_types_h.cudaGraphNode_t;
      pDependencies : System.Address;
      pNumDependencies : access corecrt_h.size_t) return driver_types_h.cudaError_t  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/cuda_runtime_api.h:9154
   with Import => True, 
        Convention => C, 
        External_Name => "cudaGraphNodeGetDependencies";

   function cudaGraphNodeGetDependentNodes
     (node : driver_types_h.cudaGraphNode_t;
      pDependentNodes : System.Address;
      pNumDependentNodes : access corecrt_h.size_t) return driver_types_h.cudaError_t  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/cuda_runtime_api.h:9186
   with Import => True, 
        Convention => C, 
        External_Name => "cudaGraphNodeGetDependentNodes";

   function cudaGraphAddDependencies
     (graph : driver_types_h.cudaGraph_t;
      from : System.Address;
      to : System.Address;
      numDependencies : corecrt_h.size_t) return driver_types_h.cudaError_t  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/cuda_runtime_api.h:9217
   with Import => True, 
        Convention => C, 
        External_Name => "cudaGraphAddDependencies";

   function cudaGraphRemoveDependencies
     (graph : driver_types_h.cudaGraph_t;
      from : System.Address;
      to : System.Address;
      numDependencies : corecrt_h.size_t) return driver_types_h.cudaError_t  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/cuda_runtime_api.h:9248
   with Import => True, 
        Convention => C, 
        External_Name => "cudaGraphRemoveDependencies";

   function cudaGraphDestroyNode (node : driver_types_h.cudaGraphNode_t) return driver_types_h.cudaError_t  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/cuda_runtime_api.h:9274
   with Import => True, 
        Convention => C, 
        External_Name => "cudaGraphDestroyNode";

   function cudaGraphInstantiate
     (pGraphExec : System.Address;
      graph : driver_types_h.cudaGraph_t;
      pErrorNode : System.Address;
      pLogBuffer : Interfaces.C.Strings.chars_ptr;
      bufferSize : corecrt_h.size_t) return driver_types_h.cudaError_t  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/cuda_runtime_api.h:9310
   with Import => True, 
        Convention => C, 
        External_Name => "cudaGraphInstantiate";

   function cudaGraphExecKernelNodeSetParams
     (hGraphExec : driver_types_h.cudaGraphExec_t;
      node : driver_types_h.cudaGraphNode_t;
      pNodeParams : access constant driver_types_h.cudaKernelNodeParams) return driver_types_h.cudaError_t  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/cuda_runtime_api.h:9344
   with Import => True, 
        Convention => C, 
        External_Name => "cudaGraphExecKernelNodeSetParams";

   function cudaGraphExecMemcpyNodeSetParams
     (hGraphExec : driver_types_h.cudaGraphExec_t;
      node : driver_types_h.cudaGraphNode_t;
      pNodeParams : access constant driver_types_h.cudaMemcpy3DParms) return driver_types_h.cudaError_t  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/cuda_runtime_api.h:9385
   with Import => True, 
        Convention => C, 
        External_Name => "cudaGraphExecMemcpyNodeSetParams";

   function cudaGraphExecMemsetNodeSetParams
     (hGraphExec : driver_types_h.cudaGraphExec_t;
      node : driver_types_h.cudaGraphNode_t;
      pNodeParams : access constant driver_types_h.cudaMemsetParams) return driver_types_h.cudaError_t  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/cuda_runtime_api.h:9426
   with Import => True, 
        Convention => C, 
        External_Name => "cudaGraphExecMemsetNodeSetParams";

   function cudaGraphExecHostNodeSetParams
     (hGraphExec : driver_types_h.cudaGraphExec_t;
      node : driver_types_h.cudaGraphNode_t;
      pNodeParams : access constant driver_types_h.cudaHostNodeParams) return driver_types_h.cudaError_t  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/cuda_runtime_api.h:9459
   with Import => True, 
        Convention => C, 
        External_Name => "cudaGraphExecHostNodeSetParams";

   function cudaGraphExecUpdate
     (hGraphExec : driver_types_h.cudaGraphExec_t;
      hGraph : driver_types_h.cudaGraph_t;
      hErrorNode_out : System.Address;
      updateResult_out : access driver_types_h.cudaGraphExecUpdateResult) return driver_types_h.cudaError_t  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/cuda_runtime_api.h:9534
   with Import => True, 
        Convention => C, 
        External_Name => "cudaGraphExecUpdate";

   function cudaGraphLaunch (graphExec : driver_types_h.cudaGraphExec_t; stream : driver_types_h.cudaStream_t) return driver_types_h.cudaError_t  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/cuda_runtime_api.h:9559
   with Import => True, 
        Convention => C, 
        External_Name => "cudaGraphLaunch";

   function cudaGraphExecDestroy (graphExec : driver_types_h.cudaGraphExec_t) return driver_types_h.cudaError_t  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/cuda_runtime_api.h:9580
   with Import => True, 
        Convention => C, 
        External_Name => "cudaGraphExecDestroy";

   function cudaGraphDestroy (graph : driver_types_h.cudaGraph_t) return driver_types_h.cudaError_t  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/cuda_runtime_api.h:9600
   with Import => True, 
        Convention => C, 
        External_Name => "cudaGraphDestroy";

   function cudaGetExportTable (ppExportTable : System.Address; pExportTableId : access constant driver_types_h.cudaUUID_t) return driver_types_h.cudaError_t  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/cuda_runtime_api.h:9605
   with Import => True, 
        Convention => C, 
        External_Name => "cudaGetExportTable";

end cuda_runtime_api_h;
