pragma Ada_2012;
pragma Style_Checks (Off);

with Interfaces.C; use Interfaces.C;
with driver_types_h;
with Interfaces.C.Extensions;

package surface_types_h is

   cudaSurfaceType1D : constant := 16#01#;  --  /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/surface_types.h:73
   cudaSurfaceType2D : constant := 16#02#;  --  /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/surface_types.h:74
   cudaSurfaceType3D : constant := 16#03#;  --  /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/surface_types.h:75
   cudaSurfaceTypeCubemap : constant := 16#0C#;  --  /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/surface_types.h:76
   cudaSurfaceType1DLayered : constant := 16#F1#;  --  /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/surface_types.h:77
   cudaSurfaceType2DLayered : constant := 16#F2#;  --  /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/surface_types.h:78
   cudaSurfaceTypeCubemapLayered : constant := 16#FC#;  --  /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/surface_types.h:79

   type cudaSurfaceBoundaryMode is 
     (cudaBoundaryModeZero,
      cudaBoundaryModeClamp,
      cudaBoundaryModeTrap)
   with Convention => C;  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/surface_types.h:84

   type cudaSurfaceFormatMode is 
     (cudaFormatModeForced,
      cudaFormatModeAuto)
   with Convention => C;  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/surface_types.h:94

   type surfaceReference is record
      channelDesc : aliased driver_types_h.cudaChannelFormatDesc;  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/surface_types.h:108
   end record
   with Convention => C_Pass_By_Copy;  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/surface_types.h:103

   subtype cudaSurfaceObject_t is Extensions.unsigned_long_long;  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/surface_types.h:114

end surface_types_h;
