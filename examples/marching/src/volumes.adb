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

package body Volumes is

   -----------
   -- Clear --
   -----------

   procedure Clear (Shape : in out Volume) is
   begin
      Shape.Halfedges.Clear;
      Shape.Faces.Clear;
      Shape.Vertices.Clear;
      Shape.Missing_Opposites.Clear;
   end Clear;

   -------------------
   -- Create_Vertex --
   -------------------

   function Create_Vertex
     (Shape : in out Volume;
      Point : Point_Real) return Volume_Index
   is
   begin
      Shape.Vertices.Append (Vertex'(Point => Point, others => <>));
      Shape.Missing_Opposites.Append (Halfedge_Index_Vectors.Empty_Vector);
      return Volume_Index (Shape.Vertices.Length - 1);
   end Create_Vertex;

   -----------------
   -- Create_Face --
   -----------------

   procedure Create_Face
     (Shape    : in out Volume;
      Indicies :        Volume_Indicies)
   is
      H : Halfedge_Vectors.Vector
        renames Shape.Halfedges;
      MO : Halfedge_Index_Vectors_Vectors.Vector
        renames Shape.Missing_Opposites;

      K                 : Integer         := 0;
      Halfedge_Indicies : Volume_Indicies := (H.Last_Index + 1,
                                              H.Last_Index + 2,
                                              H.Last_Index + 3);
   begin
      --  Errors in opposite computation ???

      Shape.Faces.Append (Halfedge_Indicies (1));

      H.Append (Halfedge'(Face   => Shape.Faces.Last_Index + 1,
                 Next   => Halfedge_Indicies (2),
                 Prev   => Halfedge_Indicies (3),
                 Vertex => Indicies (1),
                 others => <>));
      H.Append (Halfedge'(Face   => Shape.Faces.Last_Index + 1,
                 Next   => Halfedge_Indicies (3),
                 Prev   => Halfedge_Indicies (1),
                 Vertex => Indicies (2),
                 others => <>));
      H.Append (Halfedge'(Face   => Shape.Faces.Last_Index + 1,
                 Next   => Halfedge_Indicies (1),
                 Prev   => Halfedge_Indicies (2),
                 Vertex => Indicies (3),
                 others => <>));

      for J in 1..3 loop
         K := 4 - J;

         if Shape.Vertices (Indicies (J)).Halfedge = -1 then
            Shape.Vertices (Indicies (J)).Halfedge := Halfedge_Indicies (J);
         end if;

         for I in MO (Indicies (K)).First_Index ..
           MO (Indicies (K)).Last_Index
         loop
            if Halfedge_Indicies (J) /= -1 then
               --  Check if we need to connect opposites

               if H (H (MO (Indicies (K)) (I)).Prev).Vertex =
                    H (Halfedge_Indicies (J)).Vertex
                 and then H (MO (Indicies (K)) (I)).Vertex =
                            H (H (Halfedge_Indicies (J)).Prev).Vertex
               then
                  H (MO (Indicies (K)) (I)).Opposite := Halfedge_Indicies (J);
                  H (Halfedge_Indicies (J)).Opposite := MO (Indicies (K)) (I);
                  MO (Indicies (K)).Delete (I);
                  Halfedge_Indicies (J) := -1;
                  exit;
               end if;
            end if;
         end loop;

         if Halfedge_Indicies (J) /= -1 then
            MO(Indicies (J)).Append (Halfedge_Indicies (J));
         end if;
      end loop;
   end Create_Face;
end Volumes;
