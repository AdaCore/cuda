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
   type Unsigned32_Array is array (Integer range 0 .. <>) of aliased Unsigned_32;

   type Int_Access is access all Integer;

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
      XI, YI, ZI          : Integer);

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
     with CUDA_Global;

   procedure Last_Chance_Handler is null;
   pragma Export (C,
               Last_Chance_Handler,
               "__gnat_last_chance_handler");

end Marching_Cubes;
