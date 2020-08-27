with Interfaces.C; use Interfaces.C;
with CUDA.Driver_Types;
with udriver_types_h;
with Interfaces.C.Extensions;

package CUDA.Texture_Types is

   type Texture_Address_Mode is
     (Address_Mode_Wrap, Address_Mode_Clamp, Address_Mode_Mirror,
      Address_Mode_Border) with
      Convention => C;

   type Texture_Filter_Mode is (Filter_Mode_Point, Filter_Mode_Linear) with
      Convention => C;

   type Texture_Read_Mode is
     (Read_Mode_Element_Type, Read_Mode_Normalized_Float) with
      Convention => C;

   type Anon1036_Array1037 is array (0 .. 2) of Texture_Address_Mode;

   type Anon1036_Array1039 is array (0 .. 13) of int;

   type Texture_Reference is record
      Normalized                     : int;
      Filter_Mode                    : Texture_Filter_Mode;
      Address_Mode                   : Anon1036_Array1037;
      Channel_Desc                   : CUDA.Driver_Types.Channel_Format_Desc;
      S_RGB                          : int;
      Max_Anisotropy                 : unsigned;
      Mipmap_Filter_Mode             : Texture_Filter_Mode;
      Mipmap_Level_Bias              : Float;
      Min_Mipmap_Level_Clamp         : Float;
      Max_Mipmap_Level_Clamp         : Float;
      Disable_Trilinear_Optimization : int;
      Reserved                       : Anon1036_Array1039;

   end record with
      Convention => C;

   type Anon1040_Array1037 is array (0 .. 2) of Texture_Address_Mode;

   type Anon1040_Array1042 is array (0 .. 3) of Float;

   type Texture_Desc is record
      Address_Mode                   : Anon1040_Array1037;
      Filter_Mode                    : Texture_Filter_Mode;
      Read_Mode                      : Texture_Read_Mode;
      S_RGB                          : int;
      Border_Color                   : Anon1040_Array1042;
      Normalized_Coords              : int;
      Max_Anisotropy                 : unsigned;
      Mipmap_Filter_Mode             : Texture_Filter_Mode;
      Mipmap_Level_Bias              : Float;
      Min_Mipmap_Level_Clamp         : Float;
      Max_Mipmap_Level_Clamp         : Float;
      Disable_Trilinear_Optimization : int;

   end record with
      Convention => C;

   subtype Texture_Object_T is Extensions.unsigned_long_long;

end CUDA.Texture_Types;
