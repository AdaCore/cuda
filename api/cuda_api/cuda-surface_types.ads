with CUDA.Driver_Types;
with Interfaces.C;
with Interfaces.C.Extensions; use Interfaces.C;

package CUDA.Surface_Types is
   type Surface_Type is (Surface_Type1_D, Surface_Type2_D, Surface_Type3_D, Surface_Type_Cubemap, Surface_Type1_DLayered, Surface_Type2_DLayered, Surface_Type_Cubemap_Layered);
   for Surface_Type use (Surface_Type1_D => 16#01#, Surface_Type2_D => 16#02#, Surface_Type3_D => 16#03#, Surface_Type_Cubemap => 16#0C#, Surface_Type1_DLayered => 16#F1#, Surface_Type2_DLayered => 16#F2#, Surface_Type_Cubemap_Layered => 16#FC#);

   type Surface_Boundary_Mode is (Boundary_Mode_Zero, Boundary_Mode_Clamp, Boundary_Mode_Trap) with
      Convention => C;

   type Surface_Format_Mode is (Format_Mode_Forced, Format_Mode_Auto) with
      Convention => C;

   type Surface_Reference is record
      Channel_Desc : aliased CUDA.Driver_Types.Channel_Format_Desc;
   end record with
      Convention => C_Pass_By_Copy;
   subtype Surface_Object_T is Extensions.unsigned_long_long;
end CUDA.Surface_Types;
