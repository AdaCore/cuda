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

with Geometry;              use Geometry;

with Interfaces;   use Interfaces;
with Interfaces.C; use Interfaces.C;
with System; use System;

package Marching_Cubes
with SPARK_Mode => On
is
   type Unsigned32_Array is array (Integer range <>) of aliased Unsigned_32;

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
     with Pre =>
       Start.X in -2.0 ** 16 .. 2.0 ** 16
       and then Start.Y in -2.0 ** 16 .. 2.0 ** 16
       and then Start.Z in -2.0 ** 16 .. 2.0 ** 16
       and then Stop.X in -2.0 ** 16 .. 2.0 ** 16
       and then Stop.Y in -2.0 ** 16 .. 2.0 ** 16
       and then Stop.Z in -2.0 ** 16 .. 2.0 ** 16
       and then Stop.X - Start.X >= 1.0
       and then Stop.Y - Start.X >= 1.0
       and then Stop.Z - Start.X >= 1.0
       and then Lattice_Size.X in 1 .. 2 ** 8
       and then Lattice_Size.Y in 1 .. 2 ** 8
       and then Lattice_Size.Z in 1 .. 2 ** 8
       and then Last_Triangle.all >= -1
       and then Last_Vertex.all >= -1
       and then Triangles'First = 0
       and then Vertices'First = 0
       and then Triangles'Last > 0
       and then Vertices'Last > 0
       and then XI in 0 .. Lattice_Size.X - 1
       and then YI in 0 .. Lattice_Size.Y - 1
       and then ZI in 0 .. Lattice_Size.Z - 1
       and then
         (for all B of Balls => B.X in -2.0 ** 16 .. 2.0 ** 16
          and then B.X in -2.0 ** 16 .. 2.0 ** 16
          and then B.Z in -2.0 ** 16 .. 2.0 ** 16);

   procedure Mesh_CUDA
     (A_Balls             : System.Address;
      A_Triangles         : System.Address;
      A_Vertices          : System.Address;
      Ball_Size           : Integer;
      Triangles_Size      : Integer;
      Vertices_Size       : Integer;
      Start               : Point_Real;
      Stop                : Point_Real;
      Lattice_Size        : Point_Int;
      Last_Triangle       : System.Address;
      Last_Vertex         : System.Address;
      Interpolation_Steps : Positive := 4;
      Debug_Value         : System.Address)
     with SPARK_Mode => Off
   --, CUDA_Global
   ;

end Marching_Cubes;
