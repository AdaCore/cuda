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

with Geometry;              use Geometry;

with GL.Types;              use GL.Types;
with GL.Objects.Buffers;    use GL.Objects.Buffers;

with Interfaces;            use Interfaces;
with Interfaces.C.Pointers;

package Marching_Cubes is

   --------------
   -- Triangle --
   --------------

   type Triangle is record
      I1, I2, I3 : Unsigned_32 := 0;
   end record with Convention => C;

   type Triangle_Array is array (Integer range <>) of aliased Triangle;
   package Triangle_Pointers is new Interfaces.C.Pointers
     (Integer, Triangle, Triangle_Array, (others => <>));
   procedure Load_Element_Buffer is new
     GL.Objects.Buffers.Load_To_Buffer (Triangle_Pointers);

   type Unsigned32_Array is array (Integer range <>) of aliased Unsigned_32;
   package Unsigned32_Pointers is new Interfaces.C.Pointers
     (Integer, Unsigned_32, Unsigned32_Array, 0);
   procedure Load_Element_Buffer is new
     GL.Objects.Buffers.Load_To_Buffer (Unsigned32_Pointers);

   ------------
   -- Vertex --
   ------------

   type Vertex is record
      Point : Point_Real := (others => 0.0);
      Index : Integer    := 0;
   end record with Convention => C;

   type Vertex_Array is array (Integer range <>) of aliased Vertex;
   package Vertex_Pointers is new Interfaces.C.Pointers
     (Integer, Vertex, Vertex_Array, (others => <>));
   procedure Load_Element_Buffer is new
     GL.Objects.Buffers.Load_To_Buffer (Vertex_Pointers);

   ----------
   -- Mesh --
   ----------

   generic
      with function Density (Position : Point_Real) return Float;
   procedure Mesh
     (Triangles           : out Triangle_Array;
      Vertices            : out Vertex_Array;
      Start               : Point_Real;
      Stop                : Point_Real;
      Lattice_Size        : Point_Int;
      Last_Triangle       : out Integer;
      Last_Vertex         : out Integer;
      Interpolation_Steps : Positive := 4);
end Marching_Cubes;