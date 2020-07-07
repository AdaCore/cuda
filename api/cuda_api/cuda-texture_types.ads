with Interfaces.C; use Interfaces.C;
with CUDA.Driver_Types;
with driver_types_h;
with Interfaces.C.Extensions;

package CUDA.Texture_Types is
   type Texture_Address_Mode is
     (Address_Mode_Wrap, Address_Mode_Clamp, Address_Mode_Mirror,
      Address_Mode_Border);

   type Texture_Filter_Mode is (Filter_Mode_Point, Filter_Mode_Linear);

   type Texture_Read_Mode is
     (Read_Mode_Element_Type, Read_Mode_Normalized_Float);

   type Anon1082_Array1083 is array (0 .. 2) of Texture_Address_Mode;

   type Anon1082_Array1085 is array (0 .. 14) of int;

   type Texture_Reference is record
      Normalized             : int;
      Filter_Mode            : Texture_Filter_Mode;
      Address_Mode           : Anon1082_Array1083;
      Channel_Desc           : CUDA.Driver_Types.Channel_Format_Desc;
      S_RGB                  : int;
      Max_Anisotropy         : unsigned;
      Mipmap_Filter_Mode     : Texture_Filter_Mode;
      Mipmap_Level_Bias      : Float;
      Min_Mipmap_Level_Clamp : Float;
      Max_Mipmap_Level_Clamp : Float;
      Reserved               : Anon1082_Array1085;
   end record;

   type Anon1086_Array1083 is array (0 .. 2) of Texture_Address_Mode;

   type Anon1086_Array1088 is array (0 .. 3) of Float;

   type Texture_Desc is record
      Address_Mode           : Anon1086_Array1083;
      Filter_Mode            : Texture_Filter_Mode;
      Read_Mode              : Texture_Read_Mode;
      S_RGB                  : int;
      Border_Color           : Anon1086_Array1088;
      Normalized_Coords      : int;
      Max_Anisotropy         : unsigned;
      Mipmap_Filter_Mode     : Texture_Filter_Mode;
      Mipmap_Level_Bias      : Float;
      Min_Mipmap_Level_Clamp : Float;
      Max_Mipmap_Level_Clamp : Float;
   end record;
   subtype Texture_Object_T is Extensions.unsigned_long_long;
end CUDA.Texture_Types;
