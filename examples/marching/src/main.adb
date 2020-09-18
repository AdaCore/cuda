------------------------------------------------------------------------------
--                        Copyright (C) 2017, AdaCore                       --
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

with Ada.Text_IO;              use Ada.Text_IO;
with Ada.Calendar;             use Ada.Calendar;
with Ada.Directories;          use Ada.Directories;
with Interfaces.C;             use Interfaces.C;
with Interfaces;               use Interfaces;

with GL.Window;                use GL.Window;
with GL.API;                   use GL.API;
with GL.Attributes;            use GL.Attributes;
with GL.Buffers;               use GL.Buffers;
with GL.Objects.Buffers;       use GL.Objects.Buffers;
with GL.Objects.Programs;      use GL.Objects.Programs;
with GL.Objects.Shaders;       use GL.Objects.Shaders;
with GL.Objects.Vertex_Arrays; use GL.Objects.Vertex_Arrays;
with GL.Types;                 use GL.Types;
with GL.Types.Colors;          use GL.Types.Colors;
with Glfw;                     use Glfw;
with Glfw.Input;               use Glfw.Input;
with Glfw.Input.Keys;          use Glfw.Input.Keys;
with Glfw.Windows.Context;     use Glfw.Windows.Context;
with GL.Types;                 use GL.Types;
with GL.Types.Colors;          use GL.Types.Singles;
with GL.Toggles;               use GL.Toggles;
with GL.Fixed.Lighting;        use GL.Fixed.Lighting;
with GL.Uniforms;
with GL.Rasterization;

with CUDA.Runtime_Api; use CUDA.Runtime_Api;

with Maths;                    use Maths.Single_Math_Functions;
with Program_Loader;           use Program_Loader;
with Utilities;                use Utilities;

with Volumes;                  use Volumes;
with Geometry;                 use Geometry;
with Marching_Cubes;           use Marching_Cubes;
with Data;                     use Data;
with System; use System;
with CUDA.Driver_Types; use CUDA.Driver_Types;

procedure Main is        

   --  Settings and constants

   Samples      : Integer := 32;
   Max_Lattice  : constant Integer := 10;
   Scale        : constant Float   := 1.3;
   Shape        : Volume;
   Lattice_Size : Point_Int;

   type Vertex_Index_Arr is array (0 .. Samples, 0 .. Samples, 0 .. Samples, 0 .. 2) of Volume_Index;
   type Vertex_Index_Arr_Ptr is access Vertex_Index_Arr;

   Edge_Lattice : Vertex_Index_Arr_Ptr := new Vertex_Index_Arr;

   -------------------
   -- Index_To_XYZE --
   -------------------

   procedure Index_To_XYZE (I : Integer; X, Y, Z, E : out Integer) is
       xSize : Integer := Lattice_Size.X + 1;
       ySize : Integer := Lattice_Size.Y + 1;
       zSize : Integer := Lattice_Size.Z + 1;
   begin
      e := i mod 3;
      z := ((i - e) / 3) mod ySize;
      y := ((i - e - z * 3) / (3 * zSize)) mod ySize;
      x := (i - e - z * 3 - y * (3 * zSize)) / (3 * zSize * ySize);
   end Index_To_XYZE;

   ----------------------
   -- Get_Vertex_Index --
   ----------------------

   function Get_Vertex_Index (I : Unsigned_32) return Volume_Index is
      e, zi, yi, xi : Integer;
   begin
      Index_To_XYZE (Integer (i), xi, yi, zi, e);


      return R : Volume_Index := Edge_Lattice (xi, yi, zi, e) do
         if R = -1 then
            raise Program_Error;
         end if;
      end return;
   end Get_Vertex_Index;

   -------------------
   -- Create_Vertex --
   -------------------

   procedure Create_Vertex (I : Integer; P : Point_Real) is
      E, zi, yi, xi : Integer;
   begin
      Index_To_XYZE (i, xi, yi, zi, e);

      if Edge_Lattice (xi, yi, zi, e) = -1 then
         Edge_Lattice (xi, yi, zi, e) := Create_Vertex (Shape, P);
      end if;
   end Create_Vertex;

   ---------------
   -- Metaballs --
   ---------------

   --  function Metaballs (Position : Point_Real) return Float is
   --     Total : Float := 0.0;
   --     Size : constant := 0.10;
   --  begin
   --       for B of Balls loop
   --          Total := @ + Size / ((Position.x - B.x) ** 2
   --                            + (Position.y - B.y) ** 2
   --                            + (Position.Z - B.z) ** 2);
   --       end loop;
   --       return Total - 1.0;
   --  end Metaballs;

   --  Marching cubes

--   procedure Mesh_Metaballs is new Marching_Cubes.Mesh (Metaballs);

   --  Local variables

   Speeds : array (Balls'Range) of Float := (0.01, -0.02);

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
   Index_Buffer        : GL.Objects.Buffers.Buffer;
   Main_Light          : Light_Object := GL.Fixed.Lighting.Light (0);

   Last_Time         : Ada.Calendar.Time;
   Last_Clear_Time   : Ada.Calendar.Time;
   Model_View_Matrix : Singles.Matrix4;
   Temp              : Singles.Vector4;
   Temp2             : Singles.Vector3;
   --Normal_Vector     : Singles.Vector3;
   A_Vector          : Singles.Vector3;
   B_Vector          : Singles.Vector3;
   V1_Vector         : Singles.Vector3;
   V2_Vector         : Singles.Vector3;
   V3_Vector         : Singles.Vector3;
   Current_Time      : Single;
   Time_Factor       : Single;
   --J                 : Integer;
   Last_Triangle     : Integer;
   Last_Vertex       : Integer;
   FPS               : Integer := 0;
   Running           : Boolean := True;

--  Start of processing for Main

   D_Balls : System.Address;
   D_Triangles : System.Address;
   D_Vertices  : System.Address;      
   Threads_Per_Block : array (1 .. 3) of Integer := (Samples, Samples, Samples);
   Blocks_Per_Grid : Integer := 1;   
   
   procedure Load_Element_Buffer is new
     GL.Objects.Buffers.Load_To_Buffer (Triangle_Pointers);
   procedure Load_Element_Buffer is new
     GL.Objects.Buffers.Load_To_Buffer (Vertex_Pointers);
   procedure Load_Element_Buffer is new
     GL.Objects.Buffers.Load_To_Buffer (Unsigned32_Pointers);
begin   
   --  Initialize window
   
   Glfw.Init;
   Main_Window.Init (720, 480, "Marching Cubes");
   GL.Window.Set_Viewport (10, 10, 720, 480);
   --GL.Rasterization.Set_Polygon_Mode (GL.Rasterization.Line); -- Render in wireframe
   GL.Toggles.Enable (GL.Toggles.Cull_Face);
   
   --  Load shaders
   
   Put_Line (Current_Directory);
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
   --GL.Uniforms.Set_Single (Lighting_Diffuse,   Temp);
   Temp := (0.2, 0.2,  0.2, 1.0);
   --GL.Uniforms.Set_Single (Lighting_Ambient,   Temp);
   Temp := (0.5, 0.5,  0.5, 1.0);
   --GL.Uniforms.Set_Single (Lighting_Specular,  Temp);
   --GL.Uniforms.Set_Single (Lighting_Shininess, 0.2);
   Temp2 := (1.0, 1.0, 0.0);
   --GL.Uniforms.Set_Single (Lighting_Direction, Temp2);
   
   --  Main loop
   
   Last_Time       := Clock;
   Last_Clear_Time := Clock;
   
   D_Balls := Malloc (Balls'Size);
   D_Triangles := Malloc (Tris'Size);
   D_Vertices := Malloc (Verts'Size);
   
   while Running loop
      Lattice_Size := (Samples, Samples, Samples);
   
      --  Build the shapes
   
      -- CUDA program
      
      Cuda.Runtime_Api.Memcpy
        (Dst   => D_Balls,
         Src   => Balls'Address,
         Count => Balls'Size,
         Kind  => Memcpy_Host_To_Device);
      
      pragma CUDA_Execute 
        (Mesh_CUDA
           (A_Balls             => D_Balls,
            A_Triangles         => D_Triangles,
            A_Vertices          => D_Vertices,
            Start               => (-2.0, -1.0, -1.0),
            Stop                => (2.0, 1.0, 1.0),
            Lattice_Size        => Lattice_Size,
            Last_Triangle       => Last_Triangle,
            Last_Vertex         => Last_Vertex,
            Interpolation_Steps => 8),
         Threads_Per_Block, 
         Blocks_Per_Grid);
      
      Cuda.Runtime_Api.Memcpy
        (Dst   => Tris'Address,
         Src   => D_Triangles,
         Count => Tris'Size,
         Kind  => Memcpy_Device_To_Host);

      Cuda.Runtime_Api.Memcpy
        (Dst   => Verts'Address,
         Src   => D_Vertices,
         Count => Verts'Size,
         Kind  => Memcpy_Device_To_Host);
   
      Edge_Lattice.all := (others => (others => (others => (others => -1))));
   
      for V of Verts (Verts'First .. Last_Vertex) loop
         Create_Vertex (V.Index, V.Point);
      end loop;
   
      for T of Tris (Tris'First .. Last_Triangle) loop
         Create_Face (Shape,
                      (Get_Vertex_Index (T.i1),
                      Get_Vertex_Index (T.i2),
                      Get_Vertex_Index (T.i3)));
      end loop;
   
      --  Build result data
   
      declare
         Vert        : Point_Real;
         Tri         : Volume_Indicies;
         Shape_Tris  : Unsigned32_Array (0 .. (Last_Face_Index (Shape) - First_Face_Index (Shape) + 1) * 3);
         Shape_Verts : Point_Real_Array (First_Vertex_Index (Shape) .. Last_Vertex_Index (Shape));
   
         It : Integer := 0;
      begin
   
         for I in Shape_Verts'Range loop
            Vert := Get_Vertex (Shape, I);
            Shape_Verts (I) := (Vert.X * Scale, Vert.Y * Scale, Vert.Z * Scale);
         end loop;
   
         for I in First_Face_Index (Shape) .. Last_Face_Index (Shape) loop
            Tri := Get_Vertices (Shape, I);
            Shape_Tris (It) := Unsigned_32 (Tri (1));
            Shape_Tris (It + 1) := Unsigned_32 (Tri (2));
            Shape_Tris (It + 2) := Unsigned_32 (Tri (3));
   
            It := It + 3;
         end loop;
   
         --  Move the balls
   
         for I in Balls'Range loop
            if Balls (I).X + Speeds (I) not in -1.0 .. 1.0 then
               Speeds (I) := -@;
            end if;
            Balls (I).X := @ + Speeds (I);
         end loop;
   
         --  Rotate the camera
   
         Current_Time      := 0.0;
         Current_Time      := Single (Glfw.Time) / 10.0;
         Time_Factor       := Single (Samples) + 0.3 * Single (Current_Time);
         Model_View_Matrix :=
           Maths.Translation_Matrix ((0.0, 0.0, -6.0)) *
             (Maths.Rotation_Matrix (Maths.Degree (21.0 * Single (Current_Time)), (1.0, 0.0, 0.0))  *
              Maths.Rotation_Matrix (Maths.Degree (45.0 * Single (Current_Time)), (0.0, 1.0, 0.0)));
   
         --  Set shader and clear to blue and the MVP
   
         Utilities.Clear_Background_Colour ((0.0, 0.0, 0.4, 1.0));
         GL.Uniforms.Set_Single (Model_View_Location, Model_View_Matrix);
         GL.Uniforms.Set_Single (Projection_Location, Projection_Matrix);
   
         --  Update verticies
   
         Vertex_Buffer.Initialize_Id;
         Array_Buffer.Bind (Vertex_Buffer);
         Load_Element_Buffer (Array_Buffer, Shape_Verts, Static_Draw);
   
         --  Update indicies
   
         Index_Buffer.Initialize_Id;
         Element_Array_Buffer.Bind (Index_Buffer);
         Load_Element_Buffer (Element_Array_Buffer, Shape_Tris, Static_Draw);
   
         --  Calculate normal
   
         --J := Shape_Verts'First + 1;;
         --for I in Shape_Verts'Range loop
         --   if j = Shape_Verts'Last then
         --      j := Shape_Verts'First;
         --   end if;
         --   Normal_Vector (X) := Normal_Vector (X) +
         --                         (((Shape_Verts [faceVertexIndx[i]].z) + (Shape_Verts [faceVertexIndx[j]].z)) *
         --                          ((Shape_Verts [faceVertexIndx[j]].y) - (Shape_Verts [faceVertexIndx[i]].y)));
         --   Normal_Vector (Y) := Normal_Vector (Y) +
         --                         (((Shape_Verts [faceVertexIndx[i]].x) + (Shape_Verts [faceVertexIndx[j]].x)) *
         --                          ((Shape_Verts [faceVertexIndx[j]].z) - (Shape_Verts [faceVertexIndx[i]].z)));
         --   Normal_Vector (Z) := Normal_Vector (Z) +
         --                         (((Shape_Verts [faceVertexIndx[i]].y) + (Shape_Verts [faceVertexIndx[j]].y)) *
         --                          ((Shape_Verts [faceVertexIndx[j]].x) - (Shape_Verts [faceVertexIndx[i]].x)));
         --   J := J + 1;
         --end loop;
   
         V1_Vector := (Single (Shape_Verts (Shape_Verts'First).X),
                       Single (Shape_Verts (Shape_Verts'First).Y),
                       Single (Shape_Verts (Shape_Verts'First).Z));
         V2_Vector := (Single (Shape_Verts (Shape_Verts'First + 1).X),
                       Single (Shape_Verts (Shape_Verts'First + 1).Y),
                       Single (Shape_Verts (Shape_Verts'First + 1).Z));
         V3_Vector := (Single (Shape_Verts (Shape_Verts'First + 2).X),
                       Single (Shape_Verts (Shape_Verts'First + 2).Y),
                       Single (Shape_Verts (Shape_Verts'First + 2).Z));
         A_Vector  := V1_Vector - V2_Vector;
         B_Vector  := V1_Vector - V3_Vector;
         --GL.Uniforms.Set_Single (Normal_Location, GL.Types.Singles.Cross_Product (A_Vector, B_Vector));
   
         --  Render
   
         GL.Attributes.Enable_Vertex_Attrib_Array (0);
         GL.Attributes.Set_Vertex_Attrib_Pointer (0, 3, Single_Type, 0, 0);
         GL.Objects.Buffers.Draw_Elements
           (Triangles, GL.Types.Int (Last_Face_Index (Shape) - First_Face_Index (Shape) + 1) * 3, UInt_Type);
         GL.Attributes.Disable_Vertex_Attrib_Array (0);
   
         --  Rotate viewport
   
         Maths.Init_Perspective_Transform (50.0, 720.0, 480.0, 0.1, 1000.0, Projection_Matrix);
   
         --  Display FPS timing
   
         FPS := @ + 1;
         if Clock - Last_Time >= 1.0 then
            Put_Line (FPS'Image & " FPS");
            FPS       := 0;
            Last_Time := Clock;
            Last_Clear_Time := Clock;
         end if;
   
         --  Clear the buffer
   
         if Clock - Last_Clear_Time >= 0.6 then
            Last_Clear_Time := Clock;
         end if;
   
         Clear (Shape);
      end;
   
      --  Update window
   
      Glfw.Windows.Context.Swap_Buffers (Main_Window'Access);
      Glfw.Input.Poll_Events;
      Running := not (Main_Window.Should_Close
                       or Main_Window.Key_State (Escape) = Pressed);
   end loop;
   
   --  Finalize
   
   Index_Buffer.Delete_Id;
   Vertex_Array.Delete_Id;
   Vertex_Buffer.Delete_Id;
   Render_Program.Delete_Id;
   Glfw.Shutdown;
end Main;


