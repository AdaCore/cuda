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
      XI, YI, ZI          : Integer;
      Debug_Value         : not null access Interfaces.C.int)
     with Pre =>
       Start.X in - 2.0 ** 16 .. 2.0 ** 16
       and Start.Y in - 2.0 ** 16 .. 2.0 ** 16
       and Start.Z in - 2.0 ** 16 .. 2.0 ** 16
       and Stop.X in - 2.0 ** 16 .. 2.0 ** 16
       and Stop.Y in - 2.0 ** 16 .. 2.0 ** 16
       and Stop.Z in - 2.0 ** 16 .. 2.0 ** 16
       and Lattice_Size.X in 1 .. 2 ** 16
       and Lattice_Size.Y in 1 .. 2 ** 16
       and Lattice_Size.Z in 1 .. 2 ** 16
       and Last_Triangle.all >= -1
       and Last_Vertex.all >= -1
       and Triangles'First = 0
       and Vertices'First = 0
       and Triangles'Last > 0
       and Vertices'Last > 0
       and XI in 0 .. Lattice_Size.X - 1
       and YI in 0 .. Lattice_Size.Y - 1
       and ZI in 0 .. Lattice_Size.Z - 1;

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
