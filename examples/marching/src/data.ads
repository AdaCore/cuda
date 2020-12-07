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

with Marching_Cubes; use Marching_Cubes;
with Geometry; use Geometry;

package Data is

   Tris   : Triangle_Array (0 .. 1_000_000);
   Verts  : Vertex_Array (0 .. Tris'Length * 3 - 1);
   Balls  : Point_Real_Array :=
     (0 => (0.0, 0.0, 0.0),
      1 => (0.0, 0.0, 0.0),
      2 => (0.0, 0.0, 0.0),
      3 => (0.0, 0.0, 0.0),
      4 => (0.0, 0.0, 0.0));

   Speeds : array (Balls'Range) of Point_Real :=
     ((0.01, 0.0, 0.0),
      (0.0, -0.02, 0.0),
      (0.01, 0.00, 0.005),
      (0.001, 0.002, 0.0),
      (0.002, 0.0, 0.01));

   Start   : Point_Real := (-2.0, -2.0, -2.0);
   Stop    : Point_Real := (2.0, 2.0, 2.0);
   Samples : Integer := 64;
   --  Number of division for each dimensison of the space described above.


end Data;
