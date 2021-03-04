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
with SPARK_Mode => On
is

   procedure SPARK_Atomic_Add
     (Address : access int; Value : int; Old: out int; Ordering : int := 0);

   procedure SPARK_Atomic_Add
     (Address : access int; Value : int; Old: out int; Ordering : int := 0)
     with SPARK_Mode => Off
   is
   begin
      Old := Atomic_Add (Address, Value, Ordering);
   end SPARK_Atomic_Add;

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
      Last_Triangle       : not null access Interfaces.C.Int;
      Last_Vertex         : not null access Interfaces.C.Int;
      Interpolation_Steps : Positive := 4;
      XI, YI, ZI          : Integer)
     with SPARK_Mode => On
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
        with Pre =>
          (for all B of Balls => B.X in -2.0 ** 16 .. 2.0 ** 16
           and then B.Y in -2.0 ** 16 .. 2.0 ** 16
           and then B.Z in -2.0 ** 16 .. 2.0 ** 16)
           and then Position.X in -2.0 ** 16 .. 2.0 ** 18
           and then Position.Y in -2.0 ** 16 .. 2.0 ** 18
           and then Position.Z in -2.0 ** 16 .. 2.0 ** 18
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
                        Start.Z + Float (ZI) * Step.Z)) > 0.0 then 1 else 0)
          with Post => Density_Index'Result in 0 .. 1;

      --------------------
      -- Get_Edge_Index --
      --------------------

      function Get_Edge_Index
        (XI, YI, ZI : Integer;
         V1, V2     : Point_Int_01) return Unsigned_32
        with Pre =>
          Lattice_Size.X in 1 .. 2 ** 8 -- TODO: why do we need these lattice size?
          and then Lattice_Size.Y in 1 .. 2 ** 8
          and then Lattice_Size.Z in 1 .. 2 ** 8
          and then XI in 0 .. Lattice_Size.X - 1
          and then YI in 0 .. Lattice_Size.Y - 1
          and then ZI in 0 .. Lattice_Size.Z - 1
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
            raise Program_Error;
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
        with Pre => Vertices'Last > 0 -- TODO: we should not need that
        and then E in 0 .. 11
        and then Lattice_Size.X in 1 .. 2 ** 8
        and then Lattice_Size.Y in 1 .. 2 ** 8
        and then Lattice_Size.Z in 1 .. 2 ** 8
        and then Start.X in -2.0 ** 16 .. 2.0 ** 16
        and then Start.Y in -2.0 ** 16 .. 2.0 ** 16
        and then Start.Z in -2.0 ** 16 .. 2.0 ** 16
        and then Stop.X in -2.0 ** 16 .. 2.0 ** 16
        and then Stop.Y in -2.0 ** 16 .. 2.0 ** 16
        and then Stop.Z in -2.0 ** 16 .. 2.0 ** 16
        and then Stop.X - Start.X >= 1.0
        and then Stop.Y - Start.X >= 1.0
        and then Stop.Z - Start.X >= 1.0
        and then Step.X in -2.0 ** 15 .. 2.0 ** 15
        and then Step.Y in -2.0 ** 15 .. 2.0 ** 15
        and then Step.Z in -2.0 ** 15 .. 2.0 ** 15
        and then V0.X in -2.0 ** 16 .. 2.0 ** 16
        and then V0.Y in -2.0 ** 16 .. 2.0 ** 16
        and then V0.Z in -2.0 ** 16 .. 2.0 ** 16
      is
         Factor      : Float      := 0.5;
         Intr_Step   : Float;
         Dir, P1, P2 : Point_Real;
      begin
         if Record_All_Vertices
           or else V1 (E) = (0, 0, 0)
           or else V2 (E) = (0, 0, 0)
         then
            SPARK_Atomic_Add (Last_Vertex, 1, int (Vertex_Index));

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

            --  pragma Assert (V0.X in -2.0 ** 16 .. 2.0 ** 16);
            --  pragma Assert (Float (V1 (E).X) in 0.0 .. 1.0);
            --  pragma Assert (Stop.X - Start.X >= 1.0);
            --  pragma Assert (Stop.X - Start.X <= 2.0 ** 18);
            --  pragma Assert (Lattice_Size.X in 1 .. 2 ** 8);
            --  pragma Assert (Step.X in -2.0 ** 15 .. 2.0 ** 15);
            --  pragma Assert (Float (V1 (E).X) * Step.X in -2.0 ** 15 .. 2.0 ** 15);
            pragma Assert (V0.X + Float (V1 (E).X) * Step.X in -2.0 ** 17 .. 2.0 ** 17);
            --pragma Assert (Step.X in -2.0 .. 19

            pragma Assert (P1.X in -2.0 ** 17 .. 2.0 ** 17);
            pragma Assert (P1.Y in -2.0 ** 17 .. 2.0 ** 17);
            pragma Assert (P1.Z in -2.0 ** 17 .. 2.0 ** 17);

            --  Interpolate

            Intr_Step := (if Metaballs (P1) > 0.0 then -0.25 else 0.25);
            Dir  := P2 - P1;

            for I in 1 .. Interpolation_Steps loop
               pragma Loop_Invariant (Intr_Step in -0.25 .. 0.25);

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
         SPARK_Atomic_Add (Last_Triangle, 1, int (Triangle_Index));

         --  exit when Triangle_Index not in
         --    Triangles'First .. Triangles'Last - 1;

         Triangle_Index := Triangle_Index + 1;

         E0 := Triangle_Table (Index * 15 + I * 3);
         E1 := Triangle_Table (Index * 15 + I * 3 + 1);
         E2 := Triangle_Table (Index * 15 + I * 3 + 2);

         --  TODO: If using constants, we should be able to remove the
         --  assumption on the higher bound
         pragma Assume
           (E0 in 0 .. 11
            and then E1 in 0 .. 11
            and then E2 in 0 .. 11);

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
     (D_Balls             : Point_Real_Wrappers.Array_Access;
      D_Triangles         : Triangle_Wrappers.Array_Access;
      D_Vertices          : Vertex_Wrappers.Array_Access;
      Ball_Size           : Integer;
      Triangles_Size      : Integer;
      Vertices_Size       : Integer;
      Start               : Point_Real;
      Stop                : Point_Real;
      Lattice_Size        : Point_Int;
      Last_Triangle       : W_Int.T_Access;
      Last_Vertex         : W_Int.T_Access;
      Interpolation_Steps : Positive := 4;
      Debug_Value         : W_Int.T_Access)
     with SPARK_Mode => Off
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
