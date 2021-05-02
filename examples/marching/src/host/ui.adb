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

with Geometry; use Geometry;
with Marching_Cubes; use Marching_Cubes;
with Utilities; use Utilities;
with Data; use Data;

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

   procedure Render_Shape (Verts : Vertex_Array; Tris : Triangle_Array);

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
--         Albedo : GL.Uniforms.Uniform := GL.Objects.Programs.Uniform_Location (Shader, "albedo");
         Ao : GL.Uniforms.Uniform := GL.Objects.Programs.Uniform_Location (Shader, "ao");
      begin
--         GL.Uniforms.Set_Single (Albedo, 1.0, 1.0, 1.0);
         GL.Uniforms.Set_Single (Ao, 1.0);
      end;
   end Initialize;

   procedure Finalize is
   begin
      null;
   end Finalize;

   Angle : Degree := 0.0;

   procedure Draw (Verts : Vertex_Array; Tris : Triangle_Array; Running : out Boolean) is
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

      --Angle := Angle + 0.1;

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

      GL.Uniforms.Set_Single (S_Metallic, 0.9);
      GL.Uniforms.Set_Single (S_Roughtness, 0.5);
      Render_Shape (Verts, Tris);

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

   procedure Load_To_Int_Buffer is new Load_To_Buffer (Int_Pointers);
   procedure Load_To_Single_Buffer is new Load_To_Buffer (Single_Pointers);

   Scale        : constant Float   := 1.3;
   Sphere_VAO : Vertex_Array_Object;
   Vbo, Ebo : Buffer;
   Data : Single_Array(0 .. Gl.Types.Size (Samples ** 3));

   procedure Render_Shape (Verts : Vertex_Array; Tris : Triangle_Array) is
      Max_Vertex : Integer := -1;
      Vert, Norm  : Point_Real;

      It : GL.Types.Int := 0;
      Shape_Tris : Int_Array (0 .. (Tris'Length) * 3 - 1);
   begin
      if not Sphere_VAO.Initialized then
         Sphere_VAO.Initialize_Id;
         Vbo.Initialize_Id;
         Ebo.Initialize_Id;
      end if;

      for V of Verts loop
         Max_Vertex := Max_Vertex + 1;

         Vert := V.Point;
         Norm := V.Normal;

         Data (GL.Types.Int (Max_Vertex) * 9) := Single(Vert.X * Scale);
         Data (GL.Types.Int (Max_Vertex) * 9 + 1) := Single(Vert.Y * Scale);
         Data (GL.Types.Int (Max_Vertex) * 9 + 2) := Single(Vert.Z * Scale);
         Data (GL.Types.Int (Max_Vertex) * 9 + 3) := Single(Norm.X);
         Data (GL.Types.Int (Max_Vertex) * 9 + 4) := Single(Norm.Y);
         Data (GL.Types.Int (Max_Vertex) * 9 + 5) := Single(Norm.Z);
         Data (GL.Types.Int (Max_Vertex) * 9 + 6) := Single(V.Color.R);
         Data (GL.Types.Int (Max_Vertex) * 9 + 7) := Single(V.Color.G);
         Data (GL.Types.Int (Max_Vertex) * 9 + 8) := Single(V.Color.B);
      end loop;

      It := 0;

      for T of Tris loop
         Shape_Tris (It) := Gl.Types.Int (T.i1);
         Shape_Tris (It + 1) := Gl.Types.Int (T.i2);
         Shape_Tris (It + 2) := Gl.Types.Int (T.i3);

         It := It + 3;
      end loop;

      if Data'Length > 0 and then Shape_Tris'Length > 0 then
         Sphere_VAO.Bind;

         Bind (Array_Buffer, Vbo);
         Load_To_Single_Buffer (Array_Buffer, Data (0 .. Gl.Types.Int (Max_Vertex + 1) * 9 - 1), Static_Draw);

         Bind (Element_Array_Buffer, Ebo);
         Load_To_Int_Buffer (Element_Array_Buffer, Shape_Tris, Static_Draw);

         Enable_Vertex_Attrib_Array (0);
         Set_Vertex_Attrib_Pointer (0, 3, Single_Type, 9, 0);
         Enable_Vertex_Attrib_Array (2);
         Set_Vertex_Attrib_Pointer (2, 3, Single_Type, 9, 3);
         Enable_Vertex_Attrib_Array (3);
         Set_Vertex_Attrib_Pointer (3, 3, Single_Type, 9, 6);

         Draw_Elements (Triangles, Gl.Types.Int (Tris'Length) * 3, UInt_Type);
      end if;
   end Render_Shape;

end UI;

