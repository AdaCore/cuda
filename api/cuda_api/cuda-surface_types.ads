with Interfaces.C; use Interfaces.C;
with CUDA.Driver_Types;
with udriver_types_h;
with Interfaces.C.Extensions;

package CUDA.Surface_Types is

   type Surface_Boundary_Mode is
     (Boundary_Mode_Zero, Boundary_Mode_Clamp, Boundary_Mode_Trap) with
      Convention => C;

   type Surface_Format_Mode is (Format_Mode_Forced, Format_Mode_Auto) with
      Convention => C;

   type Surface_Reference is record
      Channel_Desc : CUDA.Driver_Types.Channel_Format_Desc;

   end record with
      Convention => C;

   subtype Surface_Object_T is Extensions.unsigned_long_long;

end CUDA.Surface_Types;
