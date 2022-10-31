with Ada.Strings.Unbounded; use Ada.Strings.Unbounded;

package Parameters is

   type Execution_Device_T is (Cpu, Gpu);
   type Kernel_T is (Bilateral);

   Bad_Extension : exception;

   type User_Parameters is record
      Input_Image      : Unbounded_String;
      Kernel           : Kernel_T;
      Spatial_Stdev    : Float;
      Color_Dist_Stdev : Float;
      Device           : Execution_Device_T;
      Output_Image     : Unbounded_String;
   end record;

   procedure Set_Input_Image
     (Name   : in     Unbounded_String; Value : in Unbounded_String;
      Result : in out User_Parameters);

   procedure Set_Kernel
     (Name   : in     Unbounded_String; Value : in Unbounded_String;
      Result : in out User_Parameters);

   procedure Set_Spatial_Stdev
     (Name   : in     Unbounded_String; Value : in Unbounded_String;
      Result : in out User_Parameters);

   procedure Set_Color_Dist_Stdev
     (Name   : in     Unbounded_String; Value : in Unbounded_String;
      Result : in out User_Parameters);

   procedure Set_Device
     (Name   : in     Unbounded_String; Value : in Unbounded_String;
      Result : in out User_Parameters);

   procedure Set_Output_Image
     (Name   : in     Unbounded_String; Value : in Unbounded_String;
      Result : in out User_Parameters);
end Parameters;
