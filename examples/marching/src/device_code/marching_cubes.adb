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

with Data; use Data;
with Marching_Cubes.Data; use Marching_Cubes.Data;
with CUDA.Runtime_Api;    use CUDA.Runtime_Api;
with CUDA.Device_Atomic_Functions; use CUDA.Device_Atomic_Functions;
with System.Atomic_Operations.Exchange;
with Colors; use Colors;

package body Marching_Cubes
is

   -- TODO: replace with proper runtime function when available
   function Atomic_Compare_Exchange_4
     (Ptr           : Address;
      Expected      : Address;
      Desired       : Integer;
      Weak          : Boolean   := False;
      Success_Model : Integer := 5;
      Failure_Model : Integer := 5) return Boolean;
   pragma Import (Intrinsic,
                  Atomic_Compare_Exchange_4,
                  "__atomic_compare_exchange_4");

   --  function Atomic_Load_4
   --    (Ptr   : Address;
   --     Model : Integer := 5) return Integer;
   --  pragma Import (Intrinsic, Atomic_Load_4, "__atomic_load_4");

   Edge_Lattice : array (0 .. Samples, 0 .. Samples, 0 .. Samples, 0 .. 2) of aliased Integer with Volatile;

   procedure Clear_Lattice (XI : Integer) is
   begin
      for YI in Edge_Lattice'Range (2) loop
         for ZI in Edge_Lattice'Range (3) loop
            Edge_Lattice (XI, YI, ZI, 0) := -1;
            Edge_Lattice (XI, YI, ZI, 1) := -1;
            Edge_Lattice (XI, YI, ZI, 2) := -1;
         end loop;
      end loop;
   end Clear_Lattice;

   ----------
   -- Mesh --
   ----------

   procedure Mesh
     (Balls               : Ball_Array;
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
              (Position.X - B.Position.X) ** 2 +
              (Position.Y - B.Position.Y) ** 2 +
              (Position.Z - B.Position.z) ** 2;

            if Denominator < 0.00001 then
               Denominator := 0.00001;
            end if;

            Total := Total + Size / Denominator;
         end loop;
         return Total - 1.0;
      end Metaballs;

      function Metaballs_Color
        (Position : Point_Real)
         return RGB_T
      is
         Total : RGB_T := (others => 0.0);
         Denominator : Float;
         Total_Denominator : Float := 0.0;
         HSL : HSL_T;
      begin
         for B of Balls loop
            Denominator :=
              (Position.X - B.Position.X) ** 2 +
              (Position.Y - B.Position.Y) ** 2 +
              (Position.Z - B.Position.z) ** 2;

            Total := Total + B.Color * 1.0 / Denominator;
            Total_Denominator := @ + 1.0 / Denominator;
         end loop;

         Total := Total / Total_Denominator;

         HSL := RGB_To_HSL (Total);
         HSL.S := 1.0;
         HSL.L := 0.5;
         return HSL_To_RGB (HSL);
      end Metaballs_Color;


      -------------
      -- Density --
      -------------

      function Density (P : Point_Real) return Float is
        (Metaballs (P));

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

      procedure Compute_Edge
        (Vertex_Index : Integer; E : Integer)
      is
         Factor      : Float      := 0.5;
         Intr_Step   : Float;
         Dir, P1, P2 : Point_Real;
         Point : Point_Real;
      begin
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

         Point := P1 + Dir * Factor;

         declare
            D : Float := 1.0 / 10.0;
            Grad : Point_Real;
         begin
            Grad.X := Density (Point + (D, 0.0, 0.0))
              -Density (Point + (-D, 0.0, 0.0));
            Grad.Y := Density (Point + (0.0, D, 0.0))
              -Density (Point + (0.0, -D, 0.0));
            Grad.Z := Density (Point + (0.0, D, 0.0))
              -Density (Point + (0.0, -D, 0.0));

            Vertices (Vertex_Index) := (Point => Point,
                                        Normal => -Grad,
                                        Color => Metaballs_Color (Point));
            --  TODO: Normalize Grad once we have a math run-time
         end;
      end Compute_Edge;

      function Record_Edge
        (E : Integer;
         XI, YI, ZI : Integer) return Integer
      is
         P1 : Point_Int_01 := V1 (E);
         P2 : Point_Int_01 := V2 (E);
         X1 : Integer := XI + P1.X;
         Y1 : Integer := YI + P1.Y;
         Z1 : Integer := ZI + P1.Z;
         X2 : Integer := XI + P2.X;
         Y2 : Integer := YI + P2.Y;
         Z2 : Integer := ZI + P2.Z;

         Temp       : Integer;
         Edge_Index : Integer := -1;
         Y_Size     : constant Integer := Lattice_Size.Y + 1;
         Z_Size     : constant Integer := Lattice_Size.Z + 1;

         Lattice_Value : aliased Integer := -1;
         Vertex_Index : aliased Integer;
      begin
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

         --  If the vertex index has already been computed, retrieve it

         --Vertex_Index := Atomic_Load_4 (Edge_Lattice (X1, Y1, Z1, Edge_Index)'Address);
         Vertex_Index := Edge_Lattice (X1, Y1, Z1, Edge_Index);

         if Vertex_Index /= -1 then
            while Vertex_Index < 0 loop
               --Vertex_Index := Atomic_Load_4 (Edge_Lattice (X1, Y1, Z1, Edge_Index)'Address);
               Vertex_Index := Edge_Lattice (X1, Y1, Z1, Edge_Index);
            end loop;

            return Vertex_Index;
         end if;

         --  If the vertex index has not been computed, attempt at reserving
         --  it by putting -2

         if Atomic_Compare_Exchange_4
           (Ptr      => Edge_Lattice (X1, Y1, Z1, Edge_Index)'Address,
            Expected => Lattice_Value'Address,
            Desired  => -2)
         then
            Vertex_Index := Integer (Atomic_Add (Last_Vertex, 1));

            --  TODO: probably need an atomic store here
            Edge_Lattice (X1, Y1, Z1, Edge_Index) := Vertex_Index;

            Compute_Edge (Vertex_Index, E);

            return Vertex_Index;
         else
            while Lattice_Value < 0 loop
               --Lattice_Value := Atomic_Load_4 (Edge_Lattice (X1, Y1, Z1, Edge_Index)'Address);
               Lattice_Value := Edge_Lattice (X1, Y1, Z1, Edge_Index);
            end loop;

            return Lattice_Value;
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
           (I1 => Record_Edge (E0, XI, YI, ZI),
            I2 => Record_Edge (E1, XI, YI, ZI),
            I3 => Record_Edge (E2, XI, YI, ZI));
      end loop;
   end Mesh;

   procedure Mesh_CUDA
     (D_Balls             : Ball_Array_Access;
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

   procedure Clear_Lattice_CUDA
   is
   begin
      Clear_Lattice (Integer (Block_Idx.X * Block_Dim.X + Thread_Idx.X));
   end Clear_Lattice_CUDA;

end Marching_Cubes;
