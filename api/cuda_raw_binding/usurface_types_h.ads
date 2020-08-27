pragma Ada_2012;
pragma Style_Checks (Off);
pragma Warnings ("U");

with Interfaces.C; use Interfaces.C;
with udriver_types_h;
with Interfaces.C.Extensions;

package usurface_types_h is

   cudaSurfaceType1D : constant := 16#01#;  --  /usr/local/cuda/include//surface_types.h:73
   cudaSurfaceType2D : constant := 16#02#;  --  /usr/local/cuda/include//surface_types.h:74
   cudaSurfaceType3D : constant := 16#03#;  --  /usr/local/cuda/include//surface_types.h:75
   cudaSurfaceTypeCubemap : constant := 16#0C#;  --  /usr/local/cuda/include//surface_types.h:76
   cudaSurfaceType1DLayered : constant := 16#F1#;  --  /usr/local/cuda/include//surface_types.h:77
   cudaSurfaceType2DLayered : constant := 16#F2#;  --  /usr/local/cuda/include//surface_types.h:78
   cudaSurfaceTypeCubemapLayered : constant := 16#FC#;  --  /usr/local/cuda/include//surface_types.h:79

   type cudaSurfaceBoundaryMode is 
     (cudaBoundaryModeZero,
      cudaBoundaryModeClamp,
      cudaBoundaryModeTrap)
   with Convention => C;  -- /usr/local/cuda/include//surface_types.h:84

   type cudaSurfaceFormatMode is 
     (cudaFormatModeForced,
      cudaFormatModeAuto)
   with Convention => C;  -- /usr/local/cuda/include//surface_types.h:94

   type surfaceReference is record
      channelDesc : aliased udriver_types_h.cudaChannelFormatDesc;  -- /usr/local/cuda/include//surface_types.h:108
   end record
   with Convention => C_Pass_By_Copy;  -- /usr/local/cuda/include//surface_types.h:103

   subtype cudaSurfaceObject_t is Extensions.unsigned_long_long;  -- /usr/local/cuda/include//surface_types.h:114

end usurface_types_h;
