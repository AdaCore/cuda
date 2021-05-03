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

with Geometry; use Geometry;
with Colors; use Colors;

package Data is

   Tris   : Triangle_Array (0 .. 1_000_000);
   Verts  : Vertex_Array (0 .. Tris'Length * 3 - 1);

   type Ball is record
      Position : Point_Real;
      Color : RGB_T;
      Size : Float;
      Speed : Float;
   end record;

   type Ball_Array is array (Natural range <>) of Ball;
   type Ball_Array_Access is access all Ball_Array;

   Balls  : Ball_Array :=
     (0 => (Position => (0.0, 0.0, 0.0), Color => (1.0, 0.0, 0.0), Size => 0.15, Speed => 0.0025),
      1 => (Position => (0.0, 0.0, 0.0), Color => (1.0, 0.0, 0.0), Size => 0.15, Speed => 0.0025),
      2 => (Position => (0.0, 0.0, 0.0), Color => (0.0, 1.0, 0.0), Size => 0.15, Speed => 0.0025),
      3 => (Position => (0.0, 0.0, 0.0), Color => (0.0, 1.0, 0.0), Size => 0.15, Speed => 0.0025),
      4 => (Position => (0.0, 0.0, 0.0), Color => (0.0, 0.0, 1.0), Size => 0.15, Speed => 0.0025),
      5 => (Position => (0.0, 0.0, 0.0), Color => (0.0, 0.0, 1.0), Size => 0.15, Speed => 0.0025),
      6 => (Position => (0.5, 0.0, 0.0), Color => (0.0, 0.0, 0.0), Size => -0.05, Speed => 0.01),
      7 => (Position => (0.0, 0.5, 0.0), Color => (0.0, 0.0, 0.0), Size => -0.05, Speed => 0.01),
      8 => (Position => (0.0, 0.0, 0.5), Color => (0.0, 0.0, 0.0), Size => -0.05, Speed => 0.01),
      9 => (Position => (-0.5, 0.0, 0.0), Color => (0.0, 0.0, 0.0), Size => -0.05, Speed => 0.01));

   Speeds : array (Balls'Range) of Point_Real :=
     (0 => (1.0, 0.0, 0.0) * Balls (0).Speed,
      1 => (-1.0, 0.0, 0.0) * Balls (1).Speed,
      2 => (0.0, 1.0, 0.0) * Balls (2).Speed,
      3 => (0.0, -1.0, 0.0) * Balls (3).Speed,
      4 => (0.0, 0.0, 1.0) * Balls (4).Speed,
      5 => (0.0, 0.0, -1.0) * Balls (5).Speed,
      6 => (0.0, 0.0, 1.0) * Balls (6).Speed,
      7 => (0.0, 1.0, 0.0) * Balls (7).Speed,
      8 => (1.0, 0.0, 0.0) * Balls (8).Speed,
      9 => (0.0, 0.0, 1.0) * Balls (9).Speed);

   Start   : constant Point_Real := (-2.0, -2.0, -2.0);
   Stop    : constant Point_Real := (2.0, 2.0, 2.0);
   Samples : constant Integer := 256;
   --  Number of division for each dimensison of the space described above.


end Data;
