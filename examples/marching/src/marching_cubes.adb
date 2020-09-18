------------------------------------------------------------------------------
--                       Copyright (C) 2017, AdaCore                        --
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

package body Marching_Cubes is

   ----------
   -- Mesh --
   ----------

   procedure Mesh
     (Triangles           : out Triangle_Array;
      Vertices            : out Vertex_Array;
      Start               :     Point_Real;
      Stop                :     Point_Real;
      Lattice_Size        :     Point_Int;
      Last_Triangle       : out Integer;
      Last_Vertex         : out Integer;
      Interpolation_Steps :     Positive := 4)
   is
      --  Local variables

      V0                  : Point_Real := (others => <>);
      E0, E1, E2          : Integer    := 0;
      Triangle_Index      : Integer    := 0;
      Vertex_Index        : Integer    := 0;
      Index               : Integer    := 0;
      Record_All_Vertices : Boolean    := False;

      Step : Point_Real := (X => (Stop.X - Start.X) / Float (Lattice_Size.X),
                            Y => (Stop.Y - Start.Y) / Float (Lattice_Size.Y),
                            Z => (Stop.Z - Start.Z) / Float (Lattice_Size.Z));

      -------------------
      -- Density_Index --
      -------------------

      function Density_Index (XI, YI, ZI : Integer) return Integer is
        (if Density ((Start.X + Float (XI) * Step.X,
                      Start.Y + Float (YI) * Step.Y,
                      Start.Z + Float (ZI) * Step.Z)) > 0.0 then 1 else 0);

      --------------------
      -- Get_Edge_Index --
      --------------------

      function Get_Edge_Index
        (XI, YI, ZI : Integer;
         V1, V2     : Point_Int) return Unsigned_32
      is
         X1 : Integer := XI + V1.X;
         Y1 : Integer := YI + V1.Y;
         Z1 : Integer := ZI + V1.Z;
         X2 : Integer := XI + V2.X;
         Y2 : Integer := YI + V2.Y;
         Z2 : Integer := ZI + V2.Z;

         Temp       : Integer := 0;
         Edge_Index : Integer := -1;
         Y_Size     : Integer := Lattice_Size.Y + 1;
         Z_Size     : Integer := Lattice_Size.Z + 1;
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

      procedure Record_Edge (E : Integer; TI : Unsigned_32) is
         Factor      : Float      := 0.5;
         Intr_Step   : Float      := 0.0;
         Dir, P1, P2 : Point_Real := (others => <>);
      begin
         if Record_All_Vertices
           or else V1 (E) = (0, 0, 0)
           or else V2 (E) = (0, 0, 0)
         then
            --  [Q3] There's a race condition here

            Vertex_Index := Last_Vertex;
            Last_Vertex  := Last_Vertex + 1;

            P1 := V0 + (Float (V1 (E).X) * Step.X,
                        Float (V1 (E).Y) * Step.Y,
                        Float (V1 (E).Z) * Step.Z);
            P2 := V0 + (Float (V2 (E).X) * Step.X,
                        Float (V2 (E).Y) * Step.Y,
                        Float (V2 (E).Z) * Step.Z);

            --  Interpolate

            Intr_Step := (if Density (P1) > 0.0 then -0.25 else 0.25);
            Dir  := P2 - P1;

            --  [Q6] This loop is not parallelized in the initial algorithm,
            --  but maybe worth doing?

            for I in 1 .. Interpolation_Steps loop
               if Density (P1 + Dir * Factor) > 0.0 then
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
      Last_Triangle := Triangles'First;
      Last_Vertex   := Vertices'First;

      --  [Q1] This algorithm uses a lot of locals which would introduce race
      --  conditions when parallelized if globals to the loop, but would be
      --  fine if declared locally. Do we want the language to be smart enough
      --  to localize them or enforce local scope declaration?

      --  [Q4] The three loops below needs to be parallelized. What's the
      --  semantics? Do we get maximum of X * Y * Z theads?

      --  [Q7] There is no need to copy the data from Triangles and Vertices
      --  when entering the loop. However, this data needs to be located on
      --  the GPU, then copied back to the CPU after computation. How can
      --  this be specified?

      for XI in 0 .. Lattice_Size.X - 1 loop
         for YI in 0 .. Lattice_Size.Y - 1 loop
            for ZI in 0 .. Lattice_Size.Z - 1 loop
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

               --  [Q5] This loop is not parallelized in the initial
               --  algorithm, but maybe worth doing?

               for I in 0 .. Case_To_Numpolys (Index) - 1 loop
                  --  [Q2] There's an interesting problem here. In a fully
                  --  sequencial case, I would just use triangle index and
                  --  increment it at the end of the loop. Here, because I
                  --  know that this is happening in a kernel, I'm using a
                  --  counter incremented upfront. But that's not enough,
                  --  I also need this increment to be atomic:
                  --  (1) can the first problem be automatically detected?
                  --  (2) how to make the increment and return atomic?

                  Triangle_Index := Last_Triangle;
                  Last_Triangle  := Last_Triangle + 1;

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
            end loop;
         end loop;
      end loop;

      Last_Triangle := Last_Triangle - 1;
      Last_Vertex   := Last_Vertex - 1;
   end Mesh;
end Marching_Cubes;