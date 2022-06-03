with GNAT.Spitbol.Patterns; use GNAT.Spitbol.Patterns;

with Ada.Text_IO; use Ada.Text_IO;

package body Parameters is

   function "+" (Str : Unbounded_String) return String is (To_String (Str));
   function "+" (Str : String) return Unbounded_String is (To_Unbounded_String (Str));

   Image_Type : constant VString_Var := +".ppm";
   Pat        : constant Pattern     := +Image_Type;

   procedure Set_Input_Image (Name : in Unbounded_String; Value : in Unbounded_String; Result : in out User_Parameters) is
   begin
      if Match (Value, Pat) then
         Result.Input_Image := Value;
      else
         raise Bad_extension;
      end if;
   end;

   procedure Set_Kernel (Name : in Unbounded_String; Value : in Unbounded_String; Result : in out User_Parameters) is
   begin
      Result.Kernel := Kernel_T'Value (+Value);
   end;

   procedure Set_Spatial_Stdev (Name : in Unbounded_String; Value : in Unbounded_String; Result : in out User_Parameters) is
   begin
      Result.Spatial_Stdev := Float'Value (+Value);
   end;

   procedure Set_Color_Dist_Stdev (Name : in Unbounded_String; Value : in Unbounded_String; Result : in out User_Parameters) is
   begin
      Result.Color_Dist_Stdev := Float'Value (+Value);
   end;

   procedure Set_Device (Name : in Unbounded_String; Value : in Unbounded_String; Result : in out User_Parameters) is
   begin
      Result.Device := Execution_Device_T'Value (+Value);
   end;

   procedure Set_Output_Image (Name : in Unbounded_String; Value : in Unbounded_String; Result : in out User_Parameters) is
   begin
      if Value = "" then
         declare
            Input_Image : Unbounded_String := Result.Input_Image;
         begin
            if Match (Input_Image, Pat, "") then
               Result.Output_Image := Input_Image & "_" & Result.Kernel'Image & ".ppm";
            end if;
         end;
      elsif Match (Value, Pat) then
         Result.Output_Image := Value;
      else
         raise Bad_extension;
      end if;
   end;

end Parameters;
