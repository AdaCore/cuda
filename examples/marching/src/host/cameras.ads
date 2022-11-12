with GL.Types; use GL.Types;
use GL.Types.Singles;

with Maths; use Maths;

package Cameras is

   type Camera_T is record
      Position : Vector3 := (0.0, 0.0, 0.0);
      Front    : Vector3 := (0.0, 0.0, 1.0);
      Up       : Vector3 := (1.0, 0.0, 0.0);
      Right    : Vector3 := (0.0, 1.0, 0.0);
      World_Up : Vector3 := (1.0, 0.0, 0.0);

      Yaw   : Degree := 0.0;
      Pitch : Degree := 0.0;

      Movement_Speed    : Single := 0.0;
      Mouse_Sensitivity : Single := 0.0;
      Zoom              : Degree := 0.0;
   end record;

   --  // Default camera values
   Default_Yaw         : constant := -90.0;
   Default_Pitch       : constant := 0.0;
   Default_Speed       : constant := 2.5;
   Default_Sensitivity : constant := 0.1;
   Default_Zoom        : constant := 45.0;

   function Create
     (Position : Vector3 := (0.0, 0.0, 0.0); Up : Vector3 := (0.0, 1.0, 0.0);
      Yaw      : Degree  := Default_Yaw; Pitch : Degree := Default_Pitch)
      return Camera_T;

   function Get_View_Matrix (Self : Camera_T) return Matrix4;

   procedure Process_Mouse_Movement
     (Self : in out Camera_T; X_Offset, Y_Offset : Single);

   procedure Process_Mouse_Scroll (Self : in out Camera_T; Y_Offset : Single);

   procedure Update_Camera_Vectors (Self : in out Camera_T);

end Cameras;
