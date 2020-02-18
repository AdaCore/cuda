pragma Ada_2012;
pragma Style_Checks (Off);

with Interfaces.C; use Interfaces.C;
with driver_types_h;
with Interfaces.C.Extensions;

package texture_types_h is

   cudaTextureType1D : constant := 16#01#;  --  /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/texture_types.h:73
   cudaTextureType2D : constant := 16#02#;  --  /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/texture_types.h:74
   cudaTextureType3D : constant := 16#03#;  --  /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/texture_types.h:75
   cudaTextureTypeCubemap : constant := 16#0C#;  --  /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/texture_types.h:76
   cudaTextureType1DLayered : constant := 16#F1#;  --  /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/texture_types.h:77
   cudaTextureType2DLayered : constant := 16#F2#;  --  /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/texture_types.h:78
   cudaTextureTypeCubemapLayered : constant := 16#FC#;  --  /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/texture_types.h:79

   type cudaTextureAddressMode is 
     (cudaAddressModeWrap,
      cudaAddressModeClamp,
      cudaAddressModeMirror,
      cudaAddressModeBorder)
   with Convention => C;  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/texture_types.h:84

   type cudaTextureFilterMode is 
     (cudaFilterModePoint,
      cudaFilterModeLinear)
   with Convention => C;  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/texture_types.h:95

   type cudaTextureReadMode is 
     (cudaReadModeElementType,
      cudaReadModeNormalizedFloat)
   with Convention => C;  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/texture_types.h:104

   type anon1082_addressMode_array is array (0 .. 2) of aliased cudaTextureAddressMode;
   type anon1082_uu_cudaReserved_array is array (0 .. 14) of aliased int;
   type textureReference is record
      normalized : aliased int;  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/texture_types.h:118
      filterMode : aliased cudaTextureFilterMode;  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/texture_types.h:122
      addressMode : aliased anon1082_addressMode_array;  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/texture_types.h:126
      channelDesc : aliased driver_types_h.cudaChannelFormatDesc;  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/texture_types.h:130
      sRGB : aliased int;  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/texture_types.h:134
      maxAnisotropy : aliased unsigned;  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/texture_types.h:138
      mipmapFilterMode : aliased cudaTextureFilterMode;  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/texture_types.h:142
      mipmapLevelBias : aliased float;  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/texture_types.h:146
      minMipmapLevelClamp : aliased float;  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/texture_types.h:150
      maxMipmapLevelClamp : aliased float;  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/texture_types.h:154
      uu_cudaReserved : aliased anon1082_uu_cudaReserved_array;  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/texture_types.h:155
   end record
   with Convention => C_Pass_By_Copy;  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/texture_types.h:113

   type anon1086_addressMode_array is array (0 .. 2) of aliased cudaTextureAddressMode;
   type anon1086_borderColor_array is array (0 .. 3) of aliased float;
   type cudaTextureDesc is record
      addressMode : aliased anon1086_addressMode_array;  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/texture_types.h:166
      filterMode : aliased cudaTextureFilterMode;  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/texture_types.h:170
      readMode : aliased cudaTextureReadMode;  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/texture_types.h:174
      sRGB : aliased int;  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/texture_types.h:178
      borderColor : aliased anon1086_borderColor_array;  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/texture_types.h:182
      normalizedCoords : aliased int;  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/texture_types.h:186
      maxAnisotropy : aliased unsigned;  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/texture_types.h:190
      mipmapFilterMode : aliased cudaTextureFilterMode;  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/texture_types.h:194
      mipmapLevelBias : aliased float;  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/texture_types.h:198
      minMipmapLevelClamp : aliased float;  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/texture_types.h:202
      maxMipmapLevelClamp : aliased float;  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/texture_types.h:206
   end record
   with Convention => C_Pass_By_Copy;  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/texture_types.h:161

   subtype cudaTextureObject_t is Extensions.unsigned_long_long;  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/texture_types.h:212

end texture_types_h;
