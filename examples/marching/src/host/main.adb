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

with Ada.Command_Line;         use Ada.Command_Line;
with Ada.Text_IO;              use Ada.Text_IO;
with Ada.Calendar;             use Ada.Calendar;
with Ada.Directories;          use Ada.Directories;
with Ada.Numerics.Elementary_Functions; 
use Ada.Numerics.Elementary_Functions;
with Ada.Numerics.Float_Random; use Ada.Numerics.Float_Random;
with Ada.Exceptions; use Ada.Exceptions;
with Interfaces.C;             use Interfaces.C;
with Interfaces;               use Interfaces;

with CUDA.Runtime_Api; use CUDA.Runtime_Api;
with Interfaces.C.Pointers;

with Maths;                    use Maths.Single_Math_Functions;
with Program_Loader;           use Program_Loader;
with Utilities;                use Utilities;

with Geometry;                 use Geometry;
with Marching_Cubes;           use Marching_Cubes;
with Data;                     use Data;
with System; use System;
with CUDA.Driver_Types; use CUDA.Driver_Types;
with CUDA.Vector_Types; use CUDA.Vector_Types;

with UI; use UI;

with Storage_Models;
with Storage_Models.Arrays;
with Storage_Models.Objects;
with CUDA_Storage_Models;

procedure Main is         
    
   Seed : Generator;
   
   Interpolation_Steps : constant Positive := 16;
      
   Last_Triangle     : aliased Integer;
   Last_Vertex       : aliased Integer;
   
   Last_Time         : Ada.Calendar.Time;
   
   FPS               : Integer := 0;
   
   Running           : Boolean := True;
   
   package Ball_Wrappers is new CUDA_Storage_Models.Malloc_Storage_Model.Arrays 
     (Ball, Natural, Ball_Array, Ball_Array_Access);
   package Point_Real_Wrappers is new CUDA_Storage_Models.Malloc_Storage_Model.Arrays 
     (Point_Real, Natural, Point_Real_Array, Point_Real_Array_Access);
   package Triangle_Wrappers is new CUDA_Storage_Models.Malloc_Storage_Model.Arrays
     (Triangle, Natural, Triangle_Array, Triangle_Array_Access);
   package Vertex_Wrappers is new CUDA_Storage_Models.Malloc_Storage_Model.Arrays 
     (Vertex, Natural, Vertex_Array, Vertex_Array_Access);
   
   
   W_Balls             : Ball_Wrappers.Foreign_Array_Access;   
   W_Triangles         : Triangle_Wrappers.Foreign_Array_Access;
   W_Vertices          : Vertex_Wrappers.Foreign_Array_Access;
  
   use Point_Real_Wrappers;
   use Triangle_Wrappers;
   use Vertex_Wrappers;
   use Ball_Wrappers;
   
   Threads_Per_Block : constant Dim3 := (8, 4, 4); 
   Blocks_Per_Grid : constant Dim3 :=
     (unsigned (Samples) / Threads_Per_Block.X, 
      unsigned (Samples) / Threads_Per_Block.Y,
      unsigned (Samples) / Threads_Per_Block.Z);  

   package W_Int is new CUDA_Storage_Models.Malloc_Storage_Model.Objects (Integer, Int_Access);
   use W_Int;
   
   W_Last_Triangle  : W_Int.Foreign_Access := Allocate; 
   W_Last_Vertex  : W_Int.Foreign_Access := Allocate;
   W_Debug_Value  : W_Int.Foreign_Access := Allocate;

   Debug_Value : aliased Integer := 0; 
      
   task type Compute is
      entry Set_And_Go (X1, X2, Y1, Y2, Z1, Z2 : Integer);
      entry Exit_Loop;
      entry Finished;
   end Compute;
   
   task body Compute is
      X1r, X2r, Y1r, Y2r, Z1r, Z2r : Integer;
      Do_Exit : Boolean := False;
   begin
      loop
         select
            accept Set_And_Go (X1, X2, Y1, Y2, Z1, Z2 : Integer) do
               X1r := X1;
               X2r := X2;
               Y1r := Y1;
               Y2r := Y2;
               Z1r := Z1;
               Z2r := Z2;
            end;
         or
            accept Exit_Loop do
               Do_Exit := True;
            end;
         end select;
      
         exit when Do_Exit;
         
         begin
            for XI in X1r .. X2r loop
               for YI in Y1r .. Y2r loop
                  for ZI in Z1r .. Z2r loop
                     Mesh
                       (Balls               => Balls,
                        Triangles           => Tris,
                        Vertices            => Verts,
                        Start               => Start,
                        Stop                => Stop,
                        Lattice_Size        => (Samples, Samples, Samples),
                        Last_Triangle       => Last_Triangle'Access,
                        Last_Vertex         => Last_Vertex'Access,
                        Interpolation_Steps => Interpolation_Steps,
                        XI                  => XI, 
                        YI                  => YI,
                        ZI                  => ZI);
                  end loop;
               end loop;
            end loop;
         exception
            when others =>
               Put_Line ("ERROR IN TASK");
         end;
         
         accept Finished;
      end loop;
   end Compute;
   
   Compute_Tasks : array (0 .. Samples - 1) of Compute;
   
   type Mode_Type is (Mode_CUDA, Mode_Tasking, Mode_Sequential);
   
   Mode : Mode_Type := Mode_CUDA;
   
   Compute_Started : Boolean := False;
   
begin      
   if Argument_Count >= 1 then
      if Ada.Command_Line.Argument (1) = "1" then
         Mode := Mode_CUDA;
      elsif Ada.Command_Line.Argument (1) = "2" then
         Mode := Mode_Tasking;
      elsif Ada.Command_Line.Argument (1) = "3" then
         Mode := Mode_Sequential;
      end if;
   end if;        
   
   if Mode = Mode_CUDA then
      W_Triangles := Allocate (Tris'First, Tris'Last);
      W_Vertices := Allocate (Verts'First, Verts'Last);
      W_Balls := Allocate (Balls'First, Balls'Last);
   end if;
   
   Initialize;
      
   Last_Time := Clock;  
   
   while Running loop            
      Last_Triangle := Tris'First - 1;
      Last_Vertex := Verts'First - 1;
      
      if Mode = Mode_CUDA then
         if Compute_Started then
            --  Copy back data if computation has been done
      
            Assign (Debug_Value, W_Debug_Value);
            Assign (Last_Triangle, W_Last_Triangle);
            Assign (Last_Vertex, W_Last_Vertex);
      
            Assign (Tris (0 .. Last_Triangle), W_Triangles, 0, Last_Triangle);
            Assign (Verts (0 .. Last_Vertex), W_Vertices, 0, Last_Vertex);
         end if;
         
         Assign (W_Balls, Balls);
         Assign (W_Last_Triangle, 0);
         Assign (W_Last_Vertex, 0);
         Assign (W_Debug_Value, 0);
         
         pragma CUDA_Execute
           (Clear_Lattice_CUDA,
            (Blocks_Per_Grid.X, 1, 1),
            (Threads_Per_Block.X, 1, 1));
      
         pragma CUDA_Execute
           (Mesh_CUDA
              (D_Balls             => Uncheck_Convert (W_Balls),
               D_Triangles         => Uncheck_Convert (W_Triangles),
               D_Vertices          => Uncheck_Convert (W_Vertices),
               Ball_Size           => Balls'Length,
               Triangles_Size      => Tris'Length,
               Vertices_Size       => Verts'Length,
               Start               => Start,
               Stop                => Stop,
               Lattice_Size        => (Samples, Samples, Samples),
               Last_Triangle       => Uncheck_Convert (W_Last_Triangle),
               Last_Vertex         => Uncheck_Convert (W_Last_Vertex),
               Interpolation_Steps => Interpolation_Steps,
               Debug_Value         => Uncheck_Convert (W_Debug_Value)),
            Blocks_Per_Grid,
            Threads_Per_Block);
         
         Compute_Started := True;
      elsif Mode = Mode_Sequential then
         for XI in 0 .. Samples - 1 loop
            for YI in 0 .. Samples - 1 loop
               for ZI in 0 .. Samples - 1 loop
                  Mesh
                    (Balls               => Balls,
                     Triangles           => Tris,
                     Vertices            => Verts,
                     Start               => Start,
                     Stop                => Stop,
                     Lattice_Size        => (Samples, Samples, Samples),
                     Last_Triangle       => Last_Triangle'Access,
                     Last_Vertex         => Last_Vertex'Access,
                     Interpolation_Steps => Interpolation_Steps,
                     XI                  => XI,
                     YI                  => YI,
                     ZI                  => ZI);
               end loop;
            end loop;
         end loop;
      elsif Mode = Mode_Tasking then
         for XI in 0 .. Samples - 1 loop
            Compute_Tasks (XI).Set_And_Go
              (XI, XI,
               0, Samples - 1,
               0, Samples - 1);
         end loop;
      
         for XI in 0 .. Samples - 1 loop
            Compute_Tasks (XI).Finished;
         end loop;
      end if;      
      
      --  Build result data        
      
      Draw 
        (Verts (0 .. Integer (Last_Vertex)), 
         Tris (0 .. Integer (Last_Triangle)),
         Running);      
      
      --  Move the balls
   
      for I in Balls'Range loop
         declare
            New_Position : Point_Real := Balls (I).Position;
            Speed : constant := 0.01;
         begin
            New_Position.X := Balls (I).Position.X + Speeds (I).X;
            New_Position.Y := Balls (I).Position.Y + Speeds (I).Y;
            New_Position.Z := Balls (I).Position.Z + Speeds (I).Z;
      
            if Length (New_Position) > 1.0 then
               Speeds (I).X := -@ + (Random (Seed) - 0.5) * Balls (I).Speed * 0.2;
               Speeds (I).Y := -@ + (Random (Seed) - 0.5) * Balls (I).Speed * 0.2;
               Speeds (I).Z := -@ + (Random (Seed) - 0.5) * Balls (I).Speed * 0.2;      
               
               Speeds (I) := Normalize (Speeds (I)) * Balls (I).Speed;
            end if;
      
            Balls (I).Position := New_Position;
         end;
      end loop;
      
      --  Display FPS timing
      
      FPS := @ + 1;
      if Clock - Last_Time >= 1.0 then
         Put_Line (FPS'Image & " FPS");
         FPS       := 0;
         Last_Time := Clock;
      end if;
   end loop;
   
   --  Finalize
   
   Finalize;
   
   for XI in 0 .. Samples - 1 loop
      Compute_Tasks (XI).Exit_Loop;
   end loop;
   
   if Mode = Mode_CUDA then 
      Deallocate (W_Balls);
      Deallocate (W_Vertices);
      Deallocate (W_Triangles);
      Deallocate (W_Last_Vertex);
      Deallocate (W_Last_Triangle);
      Deallocate (W_Debug_Value);
   end if;
exception
   when E : others =>
      Put_Line (Ada.Exceptions.Exception_Information (E));
      
      for XI in 0 .. Samples - 1 loop
         Compute_Tasks (XI).Exit_Loop;
      end loop;  
      
      raise;
end Main;


