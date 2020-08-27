with Interfaces.C; use Interfaces.C;

package CUDA.Device_Types is

   type Round_Mode is
     (Round_Nearest, Round_Zero, Round_Pos_Inf, Round_Min_Inf) with
      Convention => C;

end CUDA.Device_Types;
