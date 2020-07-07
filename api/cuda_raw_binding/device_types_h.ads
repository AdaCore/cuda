pragma Ada_2012;
pragma Style_Checks (Off);
pragma Warnings ("U");

with Interfaces.C; use Interfaces.C;

package device_types_h is

   type cudaRoundMode is 
     (cudaRoundNearest,
      cudaRoundZero,
      cudaRoundPosInf,
      cudaRoundMinInf)
   with Convention => C;  -- /Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include/device_types.h:66

end device_types_h;
