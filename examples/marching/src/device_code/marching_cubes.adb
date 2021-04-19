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


with Marching_Cubes.Data; use Marching_Cubes.Data;
with CUDA.Runtime_Api;    use CUDA.Runtime_Api;
with CUDA.Device_Atomic_Functions; use CUDA.Device_Atomic_Functions;

package body Marching_Cubes
is

   ----------
   -- Mesh --
   ----------

   procedure Mesh
     (Balls               : Point_Real_Array;
      Triangles           : in out Triangle_Array;
      Vertices            : in out Vertex_Array;
      Start               : Point_Real;
      Stop                : Point_Real;
      Lattice_Size        : Point_Int;
      Last_Triangle       : not null access Integer;
      Last_Vertex         : not null access Integer;
      Interpolation_Steps : Positive := 4;
      XI, YI, ZI          : Integer)
   is
      --  Local variables

      V0                  : Point_Real;
      E0, E1, E2          : Integer;
      Triangle_Index      : Integer;
      Vertex_Index        : Integer := 0;
      Index               : Integer;
      Record_All_Vertices : Boolean;

      Step : constant Point_Real :=
        (X => (Stop.X - Start.X) / Float (Lattice_Size.X),
         Y => (Stop.Y - Start.Y) / Float (Lattice_Size.Y),
         Z => (Stop.Z - Start.Z) / Float (Lattice_Size.Z));

      ---------------
      -- Metaballs --
      ---------------

      function Metaballs
        (Position : Point_Real)
         return Float
      is
         Total : Float := 0.0;
         Size : constant := 0.05;
         Denominator : Float;
      begin
         for B of Balls loop
            Denominator :=
              (Position.X - B.X) ** 2 +
              (Position.Y - B.Y) ** 2 +
              (Position.Z - B.z) ** 2;

            if Denominator < 0.00001 then
               Denominator := 0.00001;
            end if;

            Total := Total + Size / Denominator;
         end loop;
         return Total - 1.0;
      end Metaballs;

      -------------------
      -- Density_Index --
      -------------------

      function Density_Index (XI, YI, ZI : Integer) return Integer is
        (if Metaballs ((Start.X + Float (XI) * Step.X,
                      Start.Y + Float (YI) * Step.Y,
                        Start.Z + Float (ZI) * Step.Z)) > 0.0 then 1 else 0);

      --------------------
      -- Get_Edge_Index --
      --------------------

      function Get_Edge_Index
        (XI, YI, ZI : Integer;
         V1, V2     : Point_Int_01) return Unsigned_32
      is
         X1 : Integer := XI + V1.X;
         Y1 : Integer := YI + V1.Y;
         Z1 : Integer := ZI + V1.Z;
         X2 : Integer := XI + V2.X;
         Y2 : Integer := YI + V2.Y;
         Z2 : Integer := ZI + V2.Z;

         Temp       : Integer;
         Edge_Index : Integer := -1;
         Y_Size     : constant Integer := Lattice_Size.Y + 1;
         Z_Size     : constant Integer := Lattice_Size.Z + 1;
      begin
         --  Swap larger values into *2

         if X2 > X1 then
            Temp := X1;
            X1   := X2;
            X2   := Temp;
         end if;

         if Y2 > Y1 then
            Temp := Y1;
            Y1   := Y2;
            Y2   := Temp;
         end if;

         if Z2 > Z1 then
            Temp := Z1;
            Z1   := Z2;
            Z2   := Temp;
         end if;

         --  Use a unique value as the edge index

         if X1 /= X2 then
            Edge_Index := 0;
         elsif Y1 /= Y2 then
            Edge_Index := 1;
         elsif Z1 /= Z2 then
            Edge_Index := 2;
         end if;

         if Edge_Index = -1 then
            return 0;
            --  raise Program_Error;
         end if;

         --  Values go from 0 to size + 1 (boundary condition)

         return Unsigned_32 (X1 * (Y_Size * Z_Size * 3) +
                             Y1 * (Z_Size * 3) +
                             Z1 * 3 + Edge_Index);
      end Get_Edge_Index;

      -----------------
      -- Record_Edge --
      -----------------

      procedure Record_Edge (E : Integer; TI : Unsigned_32)
      is
         Factor      : Float      := 0.5;
         Intr_Step   : Float;
         Dir, P1, P2 : Point_Real;
      begin
         if Record_All_Vertices
           or else V1 (E) = (0, 0, 0)
           or else V2 (E) = (0, 0, 0)
         then
            Vertex_Index := Integer (Atomic_Add (Last_Vertex, 1));

            if Vertex_Index not in Vertices'First - 1 .. Vertices'Last - 1 then
               return;
            end if;

            Vertex_Index := Vertex_Index + 1;

            P1 := (Float (V1 (E).X) * Step.X,
                   Float (V1 (E).Y) * Step.Y,
                   Float (V1 (E).Z) * Step.Z);

            P2 := (Float (V2 (E).X) * Step.X,
                   Float (V2 (E).Y) * Step.Y,
                   Float (V2 (E).Z) * Step.Z);

            P1 := @ + V0;
            P2 := @ + V0;

            --  Interpolate

            Intr_Step := (if Metaballs (P1) > 0.0 then -0.25 else 0.25);
            Dir  := P2 - P1;

            for I in 1 .. Interpolation_Steps loop
               if Metaballs (P1 + Dir * Factor) > 0.0 then
                  Factor := Factor - Intr_Step;
               else
                  Factor := Factor + Intr_Step;
               end if;

               Intr_Step := Intr_Step / 2.0;
            end loop;

            Vertices (Vertex_Index) := (Index => Integer (TI),
                                        Point => P1 + Dir * Factor);
         end if;
      end Record_Edge;

      --  Start of processing for Mesh

   begin
      Index := ((Density_Index (XI,     YI,     ZI    )      ) +
                (Density_Index (XI,     YI + 1, ZI    ) *   2) +
                (Density_Index (XI + 1, YI + 1, ZI    ) *   4) +
                (Density_Index (XI + 1, YI,     ZI    ) *   8) +
                (Density_Index (XI,     YI,     ZI + 1) *  16) +
                (Density_Index (XI,     YI + 1, ZI + 1) *  32) +
                (Density_Index (XI + 1, YI + 1, ZI + 1) *  64) +
                (Density_Index (XI + 1, YI,     ZI + 1) * 128));

      --  Set the inital vertex for the benefit of Record_Edges

      V0 := (Start.X + Float (XI) * Step.X,
             Start.Y + Float (YI) * Step.Y,
             Start.Z + Float (ZI) * Step.Z);

      Record_All_Vertices := XI = Lattice_Size.X - 1
        or else YI = Lattice_Size.Y - 1
        or else ZI = Lattice_Size.Z - 1;

      for I in 0 .. Case_To_Numpolys (Index) - 1 loop
         Triangle_Index := Integer (Atomic_Add (Last_Triangle, 1));

         Triangle_Index := Triangle_Index + 1;

         E0 := Triangle_Table (Index * 15 + I * 3);
         E1 := Triangle_Table (Index * 15 + I * 3 + 1);
         E2 := Triangle_Table (Index * 15 + I * 3 + 2);

         Triangles (Triangle_Index) :=
           (I1 => Get_Edge_Index (XI, YI, ZI, V1 (E0), V2 (E0)),
            I2 => Get_Edge_Index (XI, YI, ZI, V1 (E1), V2 (E1)),
            I3 => Get_Edge_Index (XI, YI, ZI, V1 (E2), V2 (E2)));

         --  Only record the bottom edge {0, 0, 0}. Others will be
         --  recorded by other cubes, unless for the boundary edges
         --  (identified by Record_All_Vertices).

         --  There may be a bug with which vertices can be recorded?

         Record_Edge (E0, Triangles (Triangle_Index).I1);
         Record_Edge (E1, Triangles (Triangle_Index).I2);
         Record_Edge (E2, Triangles (Triangle_Index).I3);
      end loop;
   end Mesh;

   procedure Mesh_CUDA
     (D_Balls             : Point_Real_Array_Access;
      D_Triangles         : Triangle_Array_Access;
      D_Vertices          : Vertex_Array_Access;
      Ball_Size           : Integer;
      Triangles_Size      : Integer;
      Vertices_Size       : Integer;
      Start               : Point_Real;
      Stop                : Point_Real;
      Lattice_Size        : Point_Int;
      Last_Triangle       : Int_Access;
      Last_Vertex         : Int_Access;
      Interpolation_Steps : Positive := 4;
      Debug_Value         : Int_Access)
   is
   begin
      Mesh
        (D_Balls.all,
         D_Triangles.all,
         D_Vertices.all,
         Start,
         Stop,
         Lattice_Size,
         Last_Triangle,
          Last_Vertex,
         Interpolation_Steps,
         Integer (Block_Idx.X * Block_Dim.X + Thread_Idx.X),
         Integer (Block_Idx.Y * Block_Dim.Y + Thread_Idx.Y),
         Integer (Block_Idx.Z * Block_Dim.Z + Thread_Idx.Z));
   end Mesh_CUDA;

end Marching_Cubes;