pragma Ada_2012;
pragma Style_Checks (Off);
pragma Warnings ("U");

with Interfaces.C; use Interfaces.C;

package udevice_types_h is

   type cudaRoundMode is 
     (cudaRoundNearest,
      cudaRoundZero,
      cudaRoundPosInf,
      cudaRoundMinInf)
   with Convention => C;  -- /usr/local/cuda/include//device_types.h:66

end udevice_types_h;
