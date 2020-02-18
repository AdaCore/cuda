with CUDA.Driver_Types;
with Interfaces.C;
with Interfaces.C.Extensions; use Interfaces.C;

package CUDA.Texture_Types is
   Texture_Type1_D              : constant := 16#01#;
   Texture_Type2_D              : constant := 16#02#;
   Texture_Type3_D              : constant := 16#03#;
   Texture_Type_Cubemap         : constant := 16#0C#;
   Texture_Type1_DLayered       : constant := 16#F1#;
   Texture_Type2_DLayered       : constant := 16#F2#;
   Texture_Type_Cubemap_Layered : constant := 16#FC#;

   type Texture_Address_Mode is (Address_Mode_Wrap, Address_Mode_Clamp, Address_Mode_Mirror, Address_Mode_Border) with
      Convention => C;

   type Texture_Filter_Mode is (Filter_Mode_Point, Filter_Mode_Linear) with
      Convention => C;

   type Texture_Read_Mode is (Read_Mode_Element_Type, Read_Mode_Normalized_Float) with
      Convention => C;

   type Anon1082_Address_Mode_Array is array (0 .. 2) of aliased Texture_Address_Mode;

   type Reserved_Array is array (0 .. 14) of aliased int;

   type Texture_Reference is record
      Normalized             : aliased int;
      Filter_Mode            : aliased Texture_Filter_Mode;
      Address_Mode           : aliased Anon1082_Address_Mode_Array;
      Channel_Desc           : aliased CUDA.Driver_Types.Channel_Format_Desc;
      S_RGB                  : aliased int;
      Max_Anisotropy         : aliased unsigned;
      Mipmap_Filter_Mode     : aliased Texture_Filter_Mode;
      Mipmap_Level_Bias      : aliased Float;
      Min_Mipmap_Level_Clamp : aliased Float;
      Max_Mipmap_Level_Clamp : aliased Float;
      Reserved               : aliased Reserved_Array;
   end record with
      Convention => C_Pass_By_Copy;

   type Anon1086_Address_Mode_Array is array (0 .. 2) of aliased Texture_Address_Mode;

   type Anon1086_Border_Color_Array is array (0 .. 3) of aliased Float;

   type Texture_Desc is record
      Address_Mode           : aliased Anon1086_Address_Mode_Array;
      Filter_Mode            : aliased Texture_Filter_Mode;
      Read_Mode              : aliased Texture_Read_Mode;
      S_RGB                  : aliased int;
      Border_Color           : aliased Anon1086_Border_Color_Array;
      Normalized_Coords      : aliased int;
      Max_Anisotropy         : aliased unsigned;
      Mipmap_Filter_Mode     : aliased Texture_Filter_Mode;
      Mipmap_Level_Bias      : aliased Float;
      Min_Mipmap_Level_Clamp : aliased Float;
      Max_Mipmap_Level_Clamp : aliased Float;
   end record with
      Convention => C_Pass_By_Copy;
   subtype Texture_Object_T is Extensions.unsigned_long_long;
end CUDA.Texture_Types;
