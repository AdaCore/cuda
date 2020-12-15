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

with Interfaces; use Interfaces;
with Data; use Data;

package body Shape_Management is

   Edge_Lattice : array (0 .. Samples, 0 .. Samples, 0 .. Samples, 0 .. 2) of Volume_Index;

   -------------------
   -- Index_To_XYZE --
   -------------------

   procedure Index_To_XYZE (I : Integer; X, Y, Z, E : out Integer) is
      ySize : constant Integer := Samples + 1;
      zSize : constant Integer := Samples + 1;
   begin
      e := i mod 3;
      z := ((i - e) / 3) mod ySize;
      y := ((i - e - z * 3) / (3 * zSize)) mod ySize;
      x := (i - e - z * 3 - y * (3 * zSize)) / (3 * zSize * ySize);
   end Index_To_XYZE;

   ----------------------
   -- Get_Vertex_Index --
   ----------------------

   function Get_Vertex_Index (I : Unsigned_32) return Volume_Index is
      e, zi, yi, xi : Integer;
   begin
      Index_To_XYZE (Integer (i), xi, yi, zi, e);


      return R : constant Volume_Index := Edge_Lattice (xi, yi, zi, e) do
         if R = -1 then
            raise Program_Error;
         end if;
      end return;
   end Get_Vertex_Index;

   -------------------
   -- Create_Vertex --
   -------------------

   procedure Create_Vertex (Shape : in out Volume; I : Integer; P : Point_Real) is
      E, zi, yi, xi : Integer;
   begin
      Index_To_XYZE (i, xi, yi, zi, e);

      if Edge_Lattice (xi, yi, zi, e) = -1 then
         Edge_Lattice (xi, yi, zi, e) := Create_Vertex (Shape, P);
      end if;
   end Create_Vertex;

   -------------------
   -- Create_Volume --
   -------------------

   procedure Create_Volume (Shape : in out Volume; Verts : Vertex_Array; Tris : Triangle_Array) is
   begin
      Edge_Lattice := (others => (others => (others => (others => -1))));

      for V of Verts loop
         Create_Vertex (Shape, V.Index, V.Point);
      end loop;

      for T of Tris loop
         Create_Face (Shape,
                      (Get_Vertex_Index (T.i1),
                       Get_Vertex_Index (T.i2),
                       Get_Vertex_Index (T.i3)));
      end loop;

      Compute_Normals (Shape);
   end Create_Volume;

end Shape_Management;
