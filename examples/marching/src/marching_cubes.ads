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

package Marching_Cubes is

   type Unsigned32_Array is array (Integer range <>) of aliased Unsigned_32;

   --------------
   -- Triangle --
   --------------

   type Triangle is record
      I1, I2, I3 : Unsigned_32 := 0;
   end record with Convention => C;

   type Triangle_Array is array (Integer range <>) of aliased Triangle;

   ------------
   -- Vertex --
   ------------

   type Vertex is record
      Point : Point_Real := (others => 0.0);
      Index : Integer    := 0;
   end record with Convention => C;

   type Vertex_Array is array (Integer range <>) of aliased Vertex;

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
      Last_Triangle       : access Interfaces.C.Int;
      Last_Vertex         : access Interfaces.C.Int;
      Interpolation_Steps : Positive := 4;
      XI, YI, ZI          : Integer;
      Debug_Value         : access Interfaces.C.int);

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
     with CUDA_Global;

end Marching_Cubes;
