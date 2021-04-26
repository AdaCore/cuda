with Interfaces; use Interfaces;
with Interfaces.C; use Interfaces.C;
with Interfaces.C.Pointers;

with Ada.Text_IO; use Ada.Text_IO;
with Ada.Strings; use Ada.Strings;
with Ada.Strings.Fixed; use Ada.Strings.Fixed;
with Ada.Containers.Vectors;
use Ada.Containers;

with GL.Window;                use GL.Window;
with GL.Attributes;            use GL.Attributes;
with GL.Buffers;               use GL.Buffers;
with GL.Objects.Buffers;       use GL.Objects.Buffers;
with GL.Objects.Programs;      use GL.Objects.Programs;
with GL.Objects.Shaders;       use GL.Objects.Shaders;
with GL.Objects.Vertex_Arrays; use GL.Objects.Vertex_Arrays;
with GL.Types;                 use GL.Types;
with Glfw;                     use Glfw;
with Glfw.Windows;             use Glfw.Windows;
with Glfw.Windows.Hints;       use Glfw.Windows.Hints;
with Glfw.Input;               use Glfw.Input;
with Glfw.Input.Keys;          use Glfw.Input.Keys;
with Glfw.Input.Mouse;          use Glfw.Input.Mouse;
with Glfw.Windows.Context;     use Glfw.Windows.Context;
use GL.Types.Singles;
with GL.Toggles;               use GL.Toggles;
with GL.Fixed.Lighting;        use GL.Fixed.Lighting;
with GL.Uniforms;
use GL;

with Program_Loader; use Program_Loader;
with Maths; use Maths;
use Maths.Single_Math_Functions;

with Cameras; use Cameras;
-- This is a transcription of
-- https://learnopengl.com/code_viewer_gh.php?code=src/6.pbr/1.1.lighting/lighting.cpp

with Geometry; use Geometry;
with Marching_Cubes; use Marching_Cubes;
with Utilities; use Utilities;

package body UI is

   SCR_WIDTH : constant := 1280.0;
   SCR_HEIGHT : constant := 720.0;

   Camera : Camera_T := Create ((0.0, 0.0, 5.0));

   type My_Window_Type is new Glfw.Windows.Window with record
      null;
   end record;

   overriding
   procedure Mouse_Position_Changed (Object : not null access My_Window_Type;
                                     X, Y   : Input.Mouse.Coordinate);

   overriding
   procedure Framebuffer_Size_Changed (Object : not null access My_Window_Type;
                                       Width, Height : Natural);

   overriding
   procedure Mouse_Scrolled (Object : not null access My_Window_Type;
                             X, Y   : Input.Mouse.Scroll_Offset);

   procedure Key_Changed (Object   : not null access My_Window_Type;
                          Key      : Input.Keys.Key;
                          Scancode : Input.Keys.Scancode;
                          Action   : Input.Keys.Action;
                          Mods     : Input.Keys.Modifiers);

   overriding
   procedure Framebuffer_Size_Changed (Object : not null access My_Window_Type;
                                       Width, Height : Natural) is
   begin
      GL.Window.Set_Viewport (0, 0, GL.Types.Int(Width), GL.Types.Int (Height));
   end Framebuffer_Size_Changed;

   First_Mouse : Boolean := True;
   Last_X : Float := 800.0 / 2.0;
   Last_Y : Float := 600.0 / 2.0;

   overriding
   procedure Mouse_Position_Changed (Object : not null access My_Window_Type;
                                     X, Y   : Input.Mouse.Coordinate)
   is
      --  X_Offset, Y_Offset : Float;
   begin
      null;
      --  if First_Mouse then
      --     Last_X := Float (X);
      --     Last_Y := Float (Y);
      --     First_Mouse := False;
      --  end if;
      --
      --  X_Offset := Float (X) - Last_X;
      --  Y_Offset := Last_Y - Float (Y);
      --
      --  Last_X := Float (X);
      --  Last_Y := Float (Y);
      --
      --  Process_Mouse_Movement (Camera, Single (Y_Offset), Single (X_Offset));
   end Mouse_Position_Changed;

   procedure Key_Changed (Object   : not null access My_Window_Type;
                          Key      : Input.Keys.Key;
                          Scancode : Input.Keys.Scancode;
                          Action   : Input.Keys.Action;
                          Mods     : Input.Keys.Modifiers)
   is
   begin
      if Key = Input.Keys.Left then
         Camera.Position := @ - Camera.Right * 0.1;
      elsif Key = Input.Keys.Right then
         Camera.Position := @ + Camera.Right * 0.1;
      elsif Key = Input.Keys.Up then
         Camera.Position := @ + Camera.Up * 0.1;
      elsif Key = Input.Keys.Down then
         Camera.Position := @ - Camera.Up * 0.1;
      elsif Key = Input.Keys.A then
         Camera.Pitch := Camera.Pitch - 2.0;
         Update_Camera_Vectors (Camera);
      elsif Key = Input.Keys.D then
         Camera.Pitch := Camera.Pitch + 2.0;
         Update_Camera_Vectors (Camera);
      end if;
   end Key_Changed;

   overriding
   procedure Mouse_Scrolled (Object : not null access My_Window_Type;
                             X, Y   : Input.Mouse.Scroll_Offset) is
   begin
      Process_Mouse_Scroll (Camera, Single (Y));
   end Mouse_Scrolled;

   My_Window : aliased My_Window_Type;
   Shader : GL.Objects.Programs.Program;
   Light_Positions : Vector3_Array (0 .. 3) :=
     ((10.0,  10.0, 10.0),
      (10.0,  10.0, 10.0),
      (10.0,  10.0, 10.0),
      (10.0,  10.0, 10.0));
   Light_Colors :Vector3_Array (0 .. 3) :=
     ((300.0, 300.0, 300.0),
      (300.0, 300.0, 300.0),
      (300.0, 300.0, 300.0),
      (300.0, 300.0, 300.0));

   Nr_Rows : constant := 7;
   Nr_Columns : constant := 7;
   Spacing : constant := 2.5;

   procedure Put_Line (M : Matrix4) is
   begin
      for I in M'Range (1) loop
         for J in M'Range (2) loop
            Put (M (I, J)'Img);

            if J /= Z then
               Put (", ");
            end if;
         end loop;

         New_Line;
      end loop;
   end Put_Line;

   procedure Put_Line (V : Vector3) is
   begin
      for I in V'Range loop
         Put (V (I)'Img);

         if I /= Z then
            Put (", ");
         end if;
      end loop;

      New_Line;
   end Put_Line;

   procedure Render_Shape (Shape : Volume);

   procedure Initialize is
   begin
      Glfw.Init;

      Glfw.Windows.Hints.Set_Minimum_OpenGL_Version (3, 3);
      Glfw.Windows.Hints.Set_Samples (4);
      Glfw.Windows.Hints.Set_Profile (Core_Profile);

      My_Window.Init (Glfw.Size (SCR_WIDTH), Glfw.Size (SCR_HEIGHT), "Ada CUDA");
      My_Window.Enable_Callback (Callbacks.Framebuffer_Size);
      My_Window.Enable_Callback (Callbacks.Mouse_Scroll);
      My_Window.Enable_Callback (Callbacks.Mouse_Position);
      My_Window.Enable_Callback (Callbacks.Key);

      GL.Toggles.Enable (GL.Toggles.Depth_Test);

      Shader := Program_From
        ((Src ("src/shaders/pbr.vs", Vertex_Shader),
         Src ("src/shaders/pbr.fs", Fragment_Shader)));

      GL.Objects.Programs.Use_Program (Shader);

      declare
         Albedo : GL.Uniforms.Uniform := GL.Objects.Programs.Uniform_Location (Shader, "albedo");
         Ao : GL.Uniforms.Uniform := GL.Objects.Programs.Uniform_Location (Shader, "ao");
      begin
         GL.Uniforms.Set_Single (Albedo, 0.9, 0.0, 0.0);
         GL.Uniforms.Set_Single (Ao, 1.0);
      end;
   end Initialize;

   procedure Finalize is
   begin
      null;
   end Finalize;

   Angle : Degree := 0.0;

   procedure Draw (Shape : Volume; Running : out Boolean) is
      Delta_Time : Seconds := 0.0;
      Last_Frame : Seconds := 0.0;
      Current_Frame : Seconds;
      View : Matrix4;
      S_View : GL.Uniforms.Uniform := GL.Objects.Programs.Uniform_Location (Shader, "view");
      S_CamPos : GL.Uniforms.Uniform := GL.Objects.Programs.Uniform_Location (Shader, "camPos");
      S_Metallic : GL.Uniforms.Uniform := GL.Objects.Programs.Uniform_Location (Shader, "metallic");
      S_Roughtness : GL.Uniforms.Uniform := GL.Objects.Programs.Uniform_Location (Shader, "roughness");
      S_Model : GL.Uniforms.Uniform := GL.Objects.Programs.Uniform_Location (Shader, "model");

      function Clam (Value, Min, Max : Single) return Single is
        (if Value < Min then Min elsif Value > Max then Max else Value);
   begin
      Camera.Position (Y) := Cos (Single (Radians (Angle))) * 5.0;
      Camera.Position (Z) := Sin (Single (Radians (Angle))) * 5.0;
      Camera.Pitch := Angle + 270.0;
      Update_Camera_Vectors (Camera);

      Angle := Angle + 0.5;

      Current_Frame := Glfw.Time;

      Delta_Time := Current_Frame - Last_Frame;
      Last_Frame := Current_Frame;

      Utilities.Clear_Background_Colour ((0.0, 0.0, 0.0, 1.0));
      GL.Buffers.Clear ((Depth => True, Color => True, others => False));

      GL.Objects.Programs.Use_Program (Shader);
      View := Get_View_Matrix (Camera);

      declare
         Projection   : GL.Types.Singles.Matrix4;
         Projection_Loc : GL.Uniforms.Uniform := GL.Objects.Programs.Uniform_Location (Shader, "projection");
      begin
         Maths.Init_Perspective_Transform (50.0, Single (SCR_WIDTH), Single (SCR_HEIGHT), 0.1, 100.0, Projection);
         GL.Objects.Programs.Use_Program (Shader);
         GL.Uniforms.Set_Single (Projection_Loc, Projection);
      end;

      GL.Uniforms.Set_Single (S_View, View);
      GL.Uniforms.Set_Single (S_CamPos, Camera.Position (X), Camera.Position (Y), Camera.Position (Z));

      GL.Uniforms.Set_Single
        (S_Model,
         Matrix4'((1.0, 0.0, 0.0, 0.0),
           (0.0, 1.0, 0.0, 0.0),
           (0.0, 0.0, 1.0, 0.0),
           (0.0, 0.0, 0.0, 1.0)));

      GL.Uniforms.Set_Single (S_Metallic, 0.2);
      GL.Uniforms.Set_Single (S_Roughtness, 0.5);
      Render_Shape (Shape);

      for I in Light_Positions'Range loop
         declare
            New_Pos : Vector3 := Light_Positions (I) + Vector3'(Sin (Single (Glfw.Time) * 5.0) * 5.0, 0.0, 0.0);
            S_LightPosition : GL.Uniforms.Uniform := GL.Objects.Programs.Uniform_Location (Shader, "lightPositions[" & Trim (I'Img, Both) & "]");
            S_LightColor : GL.Uniforms.Uniform := GL.Objects.Programs.Uniform_Location (Shader, "lightColors[" & Trim (I'Img, Both) & "]");
         begin
            New_Pos := Light_Positions (I);
            GL.Uniforms.Set_Single (S_LightPosition, New_Pos);
            GL.Uniforms.Set_Single (S_LightColor, Light_Colors (I));
         end;
      end loop;

      Swap_Buffers (My_Window'Access);
      Poll_Events;

      Running := not (My_Window.Should_Close
                      or My_Window.Key_State (Escape) = Pressed);
   end Draw;

   subtype Nat is GL.Types.Int range 0 .. GL.Types.Int'Last;
   package Vector_Vector3 is new Ada.Containers.Vectors (Nat, Vector3);
   use Vector_Vector3;

   package Vector_Vector2 is new Ada.Containers.Vectors (Nat, Vector2);
   use Vector_Vector2;

   type Int_Array_Ptr is access all Int_Array;

   procedure Load_To_Int_Buffer is new Load_To_Buffer (Int_Pointers);

   type Single_Array_Ptr is access all Single_Array;

   procedure Load_To_Single_Buffer is new Load_To_Buffer (Single_Pointers);

   type Buffers_T is record
      Sphere_VAO : Vertex_Array_Object;
      Vbo, Ebo : Buffer;
   end record;

   type Buffers_Ptr is access all Buffers_T;

   Scale        : constant Float   := 1.3;

   Vertex_Buffer       : GL.Objects.Buffers.Buffer;
   Normal_Buffer       : GL.Objects.Buffers.Buffer;
   Index_Buffer        : GL.Objects.Buffers.Buffer;

   package Point_Real_Pointers is new Interfaces.C.Pointers
     (Natural, Point_Real, Point_Real_Array, (others => <>));

   package Unsigned32_Pointers is new Interfaces.C.Pointers
     (Integer, Unsigned_32, Unsigned32_Array, 0);

   procedure Load_Element_Buffer is new
     GL.Objects.Buffers.Load_To_Buffer (Unsigned32_Pointers);
   procedure Load_Element_Buffer is new
     GL.Objects.Buffers.Load_To_Buffer (Point_Real_Pointers);

   procedure Render_Shape (Shape : Volume) is
      Vert, Norm  : Point_Real;
      Tri         : Volume_Indicies;

      Tris_Count : GL.Types.Int := Gl.Types.Int ((Last_Face_Index (Shape) - First_Face_Index (Shape) + 1) * 3);
      Vert_Count : GL.Types.Int := Gl.Types.Int ((Last_Vertex_Index (Shape) - First_Vertex_Index (Shape) + 1));

      Shape_Tris  : Int_Array (0 .. Tris_Count - 1);

      It : GL.Types.Int := 0;
      Stride : GL.Types.Int := (3 * 2 * 3) * (Single'Size / 8);

      New_Buffer : Buffers_Ptr := new Buffers_T;
      Data : Single_Array_Ptr;

      Triangles_Number : GL.Types.Size :=
        GL.Types.Size (Last_Face_Index (Shape) - First_Face_Index (Shape) + 1);

      Buffers : Buffers_Ptr;
   begin
      Buffers := new Buffers_T;
      Data := new Single_Array (0 .. Vert_Count * 6 - 1);

      for I in 0 .. Vert_Count - 1 loop
         Vert := Get_Vertex (Shape, First_Vertex_Index (Shape) + Integer (I));
         Norm := Get_Normal (Shape, First_Vertex_Index (Shape) + Integer (I));
         Vert := Get_Vertex (Shape, First_Vertex_Index (Shape) + Integer (I));

         Data.all (GL.Types.Int (I) * 6) := Single(Vert.X * Scale);
         Data.all (GL.Types.Int (I) * 6 + 1) := Single(Vert.Y * Scale);
         Data.all (GL.Types.Int (I) * 6 + 2) := Single(Vert.Z * Scale);
         Data.all (GL.Types.Int (I) * 6 + 3) := Single(Norm.X);
         Data.all (GL.Types.Int (I) * 6 + 4) := Single(Norm.Y);
         Data.all (GL.Types.Int (I) * 6 + 5) := Single(Norm.Z);
      end loop;

      for I in First_Face_Index (Shape) .. Last_Face_Index (Shape) loop
         Tri := Get_Vertices (Shape, I);
         Shape_Tris (It) := Gl.Types.Int (Tri (1));
         Shape_Tris (It + 1) := Gl.Types.Int (Tri (2));
         Shape_Tris (It + 2) := Gl.Types.Int (Tri (3));

         It := It + 3;
      end loop;

      if Data.all'Length > 0 and then Shape_Tris'Length > 0 then
         Buffers.Sphere_VAO.Initialize_Id;
         Buffers.Sphere_VAO.Bind;

         Buffers.Vbo.Initialize_Id;
         Bind (Array_Buffer, Buffers.Vbo);
         Load_To_Single_Buffer (Array_Buffer, Data.all, Static_Draw);

         Buffers.Ebo.Initialize_Id;
         Bind (Element_Array_Buffer, Buffers.Ebo);
         Load_To_Int_Buffer (Element_Array_Buffer, Shape_Tris, Static_Draw);

         Enable_Vertex_Attrib_Array (0);
         Set_Vertex_Attrib_Pointer (0, 3, Single_Type, 6, 0);
         Enable_Vertex_Attrib_Array (2);
         Set_Vertex_Attrib_Pointer (2, 3, Single_Type, 6, 3);

         Draw_Elements (Triangles, Gl.Types.Int (Tris_Count), UInt_Type);
      end if;
   end Render_Shape;

end UI;

