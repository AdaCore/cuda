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

with GL.Types;              use GL.Types;
with GL.Objects.Buffers;    use GL.Objects.Buffers;

with Interfaces;            use Interfaces;
with Interfaces.C.Pointers;

package Geometry is

   ----------------
   -- Point_Real --
   ----------------

   type Point_Real is record
      X, Y, Z : Float := 0.0;
   end record with Convention => C;

   function "+" (Left, Right : Point_Real) return Point_Real is
     (Left.X + Right.X, Left.Y + Right.Y, Left.Z + Right.Z);

   function "-" (Left, Right : Point_Real) return Point_Real is
     (Left.X - Right.X, Left.Y - Right.Y, Left.Z - Right.Z);

   function "*" (Left : Point_Real; Right : Float) return Point_Real is
     (Left.X * Right, Left.Y * Right, Left.Z * Right);

   function "/" (Left : Point_Real; Right : Float) return Point_Real is
     (Left.X / Right, Left.Y / Right, Left.Z / Right);

   type Point_Real_Array is array (Integer range <>) of aliased Point_Real;

   package Point_Real_Pointers is new Interfaces.C.Pointers
     (Integer, Point_Real, Point_Real_Array, (others => <>));
   procedure Load_Element_Buffer is new
     GL.Objects.Buffers.Load_To_Buffer (Point_Real_Pointers);

   ---------------
   -- Point_Int --
   ---------------

   type Point_Int is record
      X, Y, Z : Integer := 0;
   end record;

end Geometry;
