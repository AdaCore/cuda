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

with Interfaces;            use Interfaces;
with Interfaces.C.Pointers;

with Ada.Numerics.Elementary_Functions; use Ada.Numerics.Elementary_Functions;

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

   function Cross (Left, Right : Point_Real) return Point_Real is
     (Point_Real'
        (Left.Y * Right.Z - Left.Z * Right.Y,
         Left.Z * Right.X - Left.X * Right.Z,
         Left.X * Right.Y - Left.Y * Right.X));

   function Length (R : Point_Real) return Float is
      (Sqrt (R.X ** 2 + R.Y ** 2 + R.Z ** 2));


   type Point_Real_Array is array (Integer range <>) of aliased Point_Real;

   package Point_Real_Pointers is new Interfaces.C.Pointers
     (Integer, Point_Real, Point_Real_Array, (others => <>));

   ---------------
   -- Point_Int --
   ---------------

   type Point_Int is record
      X, Y, Z : Integer := 0;
   end record;

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

end Geometry;
