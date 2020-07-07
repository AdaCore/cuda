with Interfaces.C; use Interfaces.C;
with CUDA.Driver_Types;
with driver_types_h;
with Interfaces.C.Extensions;

package CUDA.Surface_Types is
   type Surface_Type is
     (Surface_Type_1D, Surface_Type_2D, Surface_Type_3D, Surface_Type_Cubemap,
      Surface_Type_1D_Layered, Surface_Type_2D_Layered,
      Surface_Type_Cubemap_Layered);
   for Surface_Type use
     (Surface_Type_1D              => 16#01#, Surface_Type_2D => 16#02#,
      Surface_Type_3D              => 16#03#, Surface_Type_Cubemap => 16#0C#,
      Surface_Type_1D_Layered      => 16#F1#, Surface_Type_2D_Layered => 16#F2#,
      Surface_Type_Cubemap_Layered => 16#FC#);

   type Surface_Boundary_Mode is
     (Boundary_Mode_Zero, Boundary_Mode_Clamp, Boundary_Mode_Trap);

   type Surface_Format_Mode is (Format_Mode_Forced, Format_Mode_Auto);

   type Surface_Reference is record
      Channel_Desc : CUDA.Driver_Types.Channel_Format_Desc;
   end record;
   subtype Surface_Object_T is Extensions.unsigned_long_long;
end CUDA.Surface_Types;
