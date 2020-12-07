------------------------------------------------------------------------------
--                    Copyright (C) 2017-2020, AdaCore                      --
-- This is free software;  you can redistribute it  and/or modify it  under --
-- terms of the  GNU General Public License as published  by the Free Soft- --
-- ware  Foundation;  either version 3,  or (at your option) any later ver- --
-- sion.  This software is distributed in the hope  that it will be useful, --
-- but WITHOUT ANY WARRANTY;  without even the implied warranty of MERCHAN- --
-- TABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public --
-- License for  more details.  You should have  received  a copy of the GNU --
-- General  Public  License  distributed  with  this  software;   see  file --
-- COPYING3.  If not, go to http://www.gnu.org/licenses for a complete copy --
-- of the license.                                                          --
------------------------------------------------------------------------------

with Interfaces;               use Interfaces;
with Interfaces.C;             use Interfaces.C;
with Interfaces.C.Pointers;

with GL.Window;                use GL.Window;
with GL.Attributes;            use GL.Attributes;
with GL.Buffers;               use GL.Buffers;
with GL.Objects.Buffers;       use GL.Objects.Buffers;
with GL.Objects.Programs;      use GL.Objects.Programs;
with GL.Objects.Shaders;       use GL.Objects.Shaders;
with GL.Objects.Vertex_Arrays; use GL.Objects.Vertex_Arrays;
with GL.Types;                 use GL.Types;
with Glfw;                     use Glfw;
with Glfw.Input;               use Glfw.Input;
with Glfw.Input.Keys;          use Glfw.Input.Keys;
with Glfw.Windows.Context;     use Glfw.Windows.Context;
   use GL.Types.Singles;
with GL.Toggles;               use GL.Toggles;
with GL.Fixed.Lighting;        use GL.Fixed.Lighting;
with GL.Uniforms;

with Program_Loader; use Program_Loader;
with Geometry;       use Geometry;
with Marching_Cubes; use Marching_Cubes;
with Maths;          use Maths;
with Utilities;      use Utilities;

package body UI is

   Main_Window         : aliased Glfw.Windows.Window;
   Projection_Matrix   : GL.Types.Singles.Matrix4 := (others => (others => <>));
   Projection_Location : GL.Uniforms.Uniform;
   Model_View_Location : GL.Uniforms.Uniform;
   Normal_Location     : GL.Uniforms.Uniform;
   Lighting_Diffuse    : GL.Uniforms.Uniform;
   Lighting_Ambient    : GL.Uniforms.Uniform;
   Lighting_Specular   : GL.Uniforms.Uniform;
   Lighting_Shininess  : GL.Uniforms.Uniform;
   Lighting_Direction  : GL.Uniforms.Uniform;
   Render_Program      : GL.Objects.Programs.Program;
   Vertex_Array        : GL.Objects.Vertex_Arrays.Vertex_Array_Object;
   Vertex_Buffer       : GL.Objects.Buffers.Buffer;
   Normal_Buffer       : GL.Objects.Buffers.Buffer;
   Index_Buffer        : GL.Objects.Buffers.Buffer;
   Main_Light          : Light_Object := GL.Fixed.Lighting.Light (0);

   Model_View_Matrix : Singles.Matrix4;
   Temp              : Singles.Vector4;
   Temp2             : Singles.Vector3;

   Scale        : constant Float   := 1.3;

   package Unsigned32_Pointers is new Interfaces.C.Pointers
     (Integer, Unsigned_32, Unsigned32_Array, 0);

   procedure Load_Element_Buffer is new
     GL.Objects.Buffers.Load_To_Buffer (Unsigned32_Pointers);
   procedure Load_Element_Buffer is new
     GL.Objects.Buffers.Load_To_Buffer (Point_Real_Pointers);

   ----------------
   -- Initialize --
   ----------------

   procedure Initialize is
   begin
      --  Initialize window

      Glfw.Init;
      Main_Window.Init (720, 480, "Marching Cubes");
      GL.Window.Set_Viewport (10, 10, 720, 480);
      GL.Toggles.Enable (GL.Toggles.Cull_Face);
      GL.Toggles.Enable (GL.Toggles.Depth_Test);
      GL.Toggles.Enable (GL.Toggles.Lighting);
      GL.Toggles.Enable (GL.Toggles.Ligh0);

      --  Load shaders

      Render_Program := Program_From
        ((Src ("src/shaders/vert.glsl", Vertex_Shader),
         Src ("src/shaders/frag.glsl", Fragment_Shader)));
      GL.Objects.Programs.Use_Program (Render_Program);
      Model_View_Location := GL.Objects.Programs.Uniform_Location (Render_Program, "m_viewModel");
      Projection_Location := GL.Objects.Programs.Uniform_Location (Render_Program, "m_pvm");
      Normal_Location     := GL.Objects.Programs.Uniform_Location (Render_Program, "m_normal");

      --  Lighting

      Lighting_Diffuse    := GL.Objects.Programs.Uniform_Location (Render_Program, "diffuse");
      Lighting_Ambient    := GL.Objects.Programs.Uniform_Location (Render_Program, "ambient");
      Lighting_Specular   := GL.Objects.Programs.Uniform_Location (Render_Program, "specular");
      Lighting_Shininess  := GL.Objects.Programs.Uniform_Location (Render_Program, "shininess");
      Lighting_Direction  := GL.Objects.Programs.Uniform_Location (Render_Program, "l_dir");
      Temp := (0.8, 0.8,  0.8, 1.0);
      GL.Uniforms.Set_Single (Lighting_Diffuse,   Temp);
      Temp := (0.2, 0.2,  0.2, 1.0);
      GL.Uniforms.Set_Single (Lighting_Ambient,   Temp);
      Temp := (0.5, 0.5,  0.5, 1.0);
      GL.Uniforms.Set_Single (Lighting_Specular,  Temp);
      GL.Uniforms.Set_Single (Lighting_Shininess, 0.2);
      Temp2 := (1.0, 1.0, 0.0);
      GL.Uniforms.Set_Single (Lighting_Direction, Temp2);
   end Initialize;

   --------------
   -- Finalize --
   --------------

   procedure Finalize is
   begin
      Index_Buffer.Delete_Id;
      Vertex_Array.Delete_Id;
      Vertex_Buffer.Delete_Id;
      Render_Program.Delete_Id;

      Glfw.Shutdown;
   end Finalize;

   ----------
   -- Draw --
   ----------

   procedure Draw (Shape : Volume; Running : out Boolean) is
      Vert, Norm  : Point_Real;
      Tri         : Volume_Indicies;
      Shape_Tris  : Unsigned32_Array (0 .. (Last_Face_Index (Shape) - First_Face_Index (Shape) + 1) * 3);
      Shape_Verts : Point_Real_Array (First_Vertex_Index (Shape) .. Last_Vertex_Index (Shape));
      Shape_Norms : Point_Real_Array (First_Vertex_Index (Shape) .. Last_Vertex_Index (Shape));

      It : Integer := 0;
   begin
      for I in Shape_Verts'Range loop
         Vert := Get_Vertex (Shape, I);
         Norm := Get_Normal (Shape, I);
         Shape_Verts (I) := (Vert.X * Scale, Vert.Y * Scale, Vert.Z * Scale);
                  Vert := Get_Vertex (Shape, I);
         Shape_Norms (I) := (Norm.X, Norm.Y, Norm.Z);
      end loop;

      for I in First_Face_Index (Shape) .. Last_Face_Index (Shape) loop
         Tri := Get_Vertices (Shape, I);
         Shape_Tris (It) := Unsigned_32 (Tri (1));
         Shape_Tris (It + 1) := Unsigned_32 (Tri (2));
         Shape_Tris (It + 2) := Unsigned_32 (Tri (3));

         It := It + 3;
      end loop;

      --  Rotate the camera

      Model_View_Matrix := Maths.Translation_Matrix ((0.0, 0.0, -6.0));

      --  Set shader and clear to blue and the MVP

      Utilities.Clear_Background_Colour ((0.0, 0.0, 0.0, 1.0));
      GL.Buffers.Clear ((Depth => True, Color => True, others => False));
      GL.Uniforms.Set_Single (Model_View_Location, Model_View_Matrix);
      GL.Uniforms.Set_Single (Projection_Location, Projection_Matrix);

      if Shape_Verts'Length > 0 and then Shape_Tris'Length > 0 then
         --  Update verticies

         Vertex_Buffer.Initialize_Id;
         Array_Buffer.Bind (Vertex_Buffer);
         Load_Element_Buffer (Array_Buffer, Shape_Verts, Static_Draw);
         GL.Attributes.Enable_Vertex_Attrib_Array (0);
         GL.Attributes.Set_Vertex_Attrib_Pointer (0, 3, Single_Type, 0, 0);

         --  Update normals

         Normal_Buffer.Initialize_Id;
         Array_Buffer.Bind (Normal_Buffer);
         Load_Element_Buffer (Array_Buffer, Shape_Norms, Static_Draw);
         GL.Attributes.Enable_Vertex_Attrib_Array (2);
         GL.Attributes.Set_Vertex_Attrib_Pointer (2, 3, Single_Type, 0, 0);

         --  Update indicies

         Index_Buffer.Initialize_Id;
         Element_Array_Buffer.Bind (Index_Buffer);
         Load_Element_Buffer (Element_Array_Buffer, Shape_Tris, Static_Draw);

         --  Render

         GL.Objects.Buffers.Draw_Elements
           (Triangles,
            GL.Types.Int (Last_Face_Index (Shape) - First_Face_Index (Shape) + 1) * 3,
            UInt_Type);

         GL.Attributes.Disable_Vertex_Attrib_Array (0);
         GL.Attributes.Disable_Vertex_Attrib_Array (2);
      end if;

      --  Rotate viewport

      declare
         Width, Height : Glfw.Size;
      begin
         Glfw.Windows.Get_Size (Main_Window'Access, Width, Height);

         GL.Window.Set_Viewport (10, 10, GL.Types.Int (Width), GL.Types.Int (Height));
         Maths.Init_Perspective_Transform (50.0, Single (Width), Single (Height), 0.1, 1000.0, Projection_Matrix);
      end;

      --  Update window

      Glfw.Windows.Context.Swap_Buffers (Main_Window'Access);
      Glfw.Input.Poll_Events;
      Running := not (Main_Window.Should_Close
                      or Main_Window.Key_State (Escape) = Pressed);
   end Draw;

end UI;
