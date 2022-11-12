with GL; use GL;

with Maths; use Maths;
use Maths.Single_Math_Functions;

package body Cameras is

   function Create
     (Position : Vector3 := (0.0, 0.0, 0.0); Up : Vector3 := (0.0, 1.0, 0.0);
      Yaw      : Degree  := Default_Yaw; Pitch : Degree := Default_Pitch)
      return Camera_T
   is
      Result : Camera_T;
   begin
      Result.Front             := (0.0, 0.0, -1.0);
      Result.Movement_Speed    := Default_Speed;
      Result.Mouse_Sensitivity := Default_Sensitivity;
      Result.Zoom              := Default_Zoom;

      Result.Position := Position;
      Result.Up       := Up;
      Result.Yaw      := Yaw;
      Result.Pitch    := Pitch;

      Update_Camera_Vectors (Result);

      return Result;
   end Create;

   function Get_View_Matrix (Self : Camera_T) return Matrix4 is
      Result : Matrix4;
   begin
      Init_Lookat_Transform
        (Self.Position, Self.Position + Self.Front, Self.Up, Result);
      return Result;
   end Get_View_Matrix;

   procedure Update_Camera_Vectors (Self : in out Camera_T) is
   begin
      Self.Front :=
        Normalized
          ((Cos (Single (Radians (Self.Yaw))) *
            Cos (Single (Radians (Self.Pitch))),
            Sin (Single (Radians (Self.Pitch))),
            Sin (Single (Radians (Self.Yaw))) *
            Cos (Single (Radians (Self.Pitch)))));

      Self.Right := Normalized (Cross_Product (Self.Front, Self.World_Up));
      Self.Up    := Normalized (Cross_Product (Self.Right, Self.Front));
   end Update_Camera_Vectors;

   procedure Process_Mouse_Movement
     (Self : in out Camera_T; X_Offset, Y_Offset : Single)
   is
   begin
      Self.Yaw   := @ + Degree (X_Offset * Self.Mouse_Sensitivity);
      Self.Pitch := @ + Degree (Y_Offset * Self.Mouse_Sensitivity);

      if Self.Pitch > 89.0 then
         Self.Pitch := 89.0;
      elsif Self.Pitch < -89.0 then
         Self.Pitch := -89.0;
      end if;

      Update_Camera_Vectors (Self);
   end Process_Mouse_Movement;

   procedure Process_Mouse_Scroll (Self : in out Camera_T; Y_Offset : Single)
   is
   begin
      --  Self.Zoom := @ - Degree (Y_Offset);
      --
      --  if Self.Zoom < 1.0 then
      --     Self.Zoom := 1.0;
      --  elsif Self.Zoom > 45.0 then
      --     Self.Zoom := 45.0;
      --  end if;

      Self.Position := @ + Self.Front * Y_Offset;

      Update_Camera_Vectors (Self);
   end Process_Mouse_Scroll;

end Cameras;
