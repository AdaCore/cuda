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

with Interfaces;                        use Interfaces;
with Interfaces.C.Pointers;
with Ada.Numerics.Elementary_Functions; use Ada.Numerics.Elementary_Functions;
with Colors;                            use Colors;

package Geometry is

   ----------------
   -- Point_Real --
   ----------------

   type Point_Real is record
      X, Y, Z : Float;
   end record with
     Convention => C;

   function "+" (Left, Right : Point_Real) return Point_Real is
     (Left.X + Right.X, Left.Y + Right.Y, Left.Z + Right.Z);

   function "-" (Left, Right : Point_Real) return Point_Real is
     (Left.X - Right.X, Left.Y - Right.Y, Left.Z - Right.Z);

   function "-" (Right : Point_Real) return Point_Real is
     (-Right.X, -Right.Y, -Right.Z);

   function "*" (Left : Point_Real; Right : Float) return Point_Real is
     (Left.X * Right, Left.Y * Right, Left.Z * Right);

   function "/" (Left : Point_Real; Right : Float) return Point_Real is
     (Left.X / Right, Left.Y / Right, Left.Z / Right);

   function Cross (Left, Right : Point_Real) return Point_Real is
     (Point_Real'
        (Left.Y * Right.Z - Left.Z * Right.Y,
         Left.Z * Right.X - Left.X * Right.Z,
         Left.X * Right.Y - Left.Y * Right.X));

   type Point_Real_Array is array (Natural range 0 .. <>) of aliased Point_Real;
   type Point_Real_Array_Access is access all Point_Real_Array;

   ---------------
   -- Point_Int --
   ---------------

   type Point_Int is record
      X, Y, Z : Integer;
   end record;

   --------------
   -- Triangle --
   --------------

   type Triangle is record
      I1, I2, I3 : Integer;
   end record;

   type Triangle_Array is array (Natural range 0 .. <>) of aliased Triangle;
   type Triangle_Array_Access is access all Triangle_Array;

   ------------
   -- Vertex --
   ------------

   type Vertex is record
      Point  : Point_Real;
      Normal : Point_Real;
      Color  : RGB_T;
   end record;

   type Vertex_Array is array (Natural range 0 .. <>) of aliased Vertex;
   type Vertex_Array_Access is access all Vertex_Array;

end Geometry;
