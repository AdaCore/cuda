pragma Ada_2012;
pragma Style_Checks (Off);
pragma Warnings ("U");

with Interfaces.C; use Interfaces.C;
with udriver_types_h;
with Interfaces.C.Extensions;

package utexture_types_h is

   cudaTextureType1D : constant := 16#01#;  --  /usr/local/cuda/include//texture_types.h:73
   cudaTextureType2D : constant := 16#02#;  --  /usr/local/cuda/include//texture_types.h:74
   cudaTextureType3D : constant := 16#03#;  --  /usr/local/cuda/include//texture_types.h:75
   cudaTextureTypeCubemap : constant := 16#0C#;  --  /usr/local/cuda/include//texture_types.h:76
   cudaTextureType1DLayered : constant := 16#F1#;  --  /usr/local/cuda/include//texture_types.h:77
   cudaTextureType2DLayered : constant := 16#F2#;  --  /usr/local/cuda/include//texture_types.h:78
   cudaTextureTypeCubemapLayered : constant := 16#FC#;  --  /usr/local/cuda/include//texture_types.h:79

   type cudaTextureAddressMode is 
     (cudaAddressModeWrap,
      cudaAddressModeClamp,
      cudaAddressModeMirror,
      cudaAddressModeBorder)
   with Convention => C;  -- /usr/local/cuda/include//texture_types.h:84

   type cudaTextureFilterMode is 
     (cudaFilterModePoint,
      cudaFilterModeLinear)
   with Convention => C;  -- /usr/local/cuda/include//texture_types.h:95

   type cudaTextureReadMode is 
     (cudaReadModeElementType,
      cudaReadModeNormalizedFloat)
   with Convention => C;  -- /usr/local/cuda/include//texture_types.h:104

   type anon1036_array1037 is array (0 .. 2) of aliased cudaTextureAddressMode;
   type anon1036_array1039 is array (0 .. 13) of aliased int;
   type textureReference is record
      normalized : aliased int;  -- /usr/local/cuda/include//texture_types.h:118
      filterMode : aliased cudaTextureFilterMode;  -- /usr/local/cuda/include//texture_types.h:122
      addressMode : aliased anon1036_array1037;  -- /usr/local/cuda/include//texture_types.h:126
      channelDesc : aliased udriver_types_h.cudaChannelFormatDesc;  -- /usr/local/cuda/include//texture_types.h:130
      sRGB : aliased int;  -- /usr/local/cuda/include//texture_types.h:134
      maxAnisotropy : aliased unsigned;  -- /usr/local/cuda/include//texture_types.h:138
      mipmapFilterMode : aliased cudaTextureFilterMode;  -- /usr/local/cuda/include//texture_types.h:142
      mipmapLevelBias : aliased float;  -- /usr/local/cuda/include//texture_types.h:146
      minMipmapLevelClamp : aliased float;  -- /usr/local/cuda/include//texture_types.h:150
      maxMipmapLevelClamp : aliased float;  -- /usr/local/cuda/include//texture_types.h:154
      disableTrilinearOptimization : aliased int;  -- /usr/local/cuda/include//texture_types.h:158
      uu_cudaReserved : aliased anon1036_array1039;  -- /usr/local/cuda/include//texture_types.h:159
   end record
   with Convention => C_Pass_By_Copy;  -- /usr/local/cuda/include//texture_types.h:113

   type anon1040_array1037 is array (0 .. 2) of aliased cudaTextureAddressMode;
   type anon1040_array1042 is array (0 .. 3) of aliased float;
   type cudaTextureDesc is record
      addressMode : aliased anon1040_array1037;  -- /usr/local/cuda/include//texture_types.h:170
      filterMode : aliased cudaTextureFilterMode;  -- /usr/local/cuda/include//texture_types.h:174
      readMode : aliased cudaTextureReadMode;  -- /usr/local/cuda/include//texture_types.h:178
      sRGB : aliased int;  -- /usr/local/cuda/include//texture_types.h:182
      borderColor : aliased anon1040_array1042;  -- /usr/local/cuda/include//texture_types.h:186
      normalizedCoords : aliased int;  -- /usr/local/cuda/include//texture_types.h:190
      maxAnisotropy : aliased unsigned;  -- /usr/local/cuda/include//texture_types.h:194
      mipmapFilterMode : aliased cudaTextureFilterMode;  -- /usr/local/cuda/include//texture_types.h:198
      mipmapLevelBias : aliased float;  -- /usr/local/cuda/include//texture_types.h:202
      minMipmapLevelClamp : aliased float;  -- /usr/local/cuda/include//texture_types.h:206
      maxMipmapLevelClamp : aliased float;  -- /usr/local/cuda/include//texture_types.h:210
      disableTrilinearOptimization : aliased int;  -- /usr/local/cuda/include//texture_types.h:214
   end record
   with Convention => C_Pass_By_Copy;  -- /usr/local/cuda/include//texture_types.h:165

   subtype cudaTextureObject_t is Extensions.unsigned_long_long;  -- /usr/local/cuda/include//texture_types.h:220

end utexture_types_h;
