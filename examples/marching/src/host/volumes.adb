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

with Ada.Numerics.Elementary_Functions; use Ada.Numerics.Elementary_Functions;

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
      Point : Point_Real;
      Normal : Point_Real) return Volume_Index
   is
   begin
      Shape.Vertices.Append (Vertex'(Point => Point, Normal => Normal, others => <>));
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


   type Operation_T is access procedure (Vert : Vertex; Forward : Boolean);

   procedure Apply_To_Vertex_Ring (Shape : in out Volume; Center : Halfedge; Operation : Operation_T) is
      Start_H : Halfedge := Center;
      It_H : Halfedge := Start_H;

      H : Halfedge_Vectors.Vector renames Shape.Halfedges;
      V : Vertex_Vectors.Vector renames Shape.Vertices;
   begin
      loop
         It_H := H (It_H.Next);
         Operation (V (It_H.Vertex), True);

         if It_H.Opposite = -1 then
            exit;
         end if;

         It_H := H (It_H.Opposite);

         if It_H = Start_H then
            --  The ring has been fully operated
            return;
         end if;
      end loop;

      -- The ring could not be completed,
      -- because of opposites missing.
      -- retreive the other side of the ring

      It_H := Start_H;

      if It_H.Prev = -1 then
         return;
      end if;

      It_H := H (It_H.Prev);

      loop
         Operation (V (It_H.Vertex), False);

         It_H := H (It_H.Next);

         if It_H.Opposite = -1 then
            exit;
         end if;

         It_H := H (It_H.Opposite);

         It_H := H (It_H.Next);

         if It_H = Start_H then
            raise Program_Error with "INCONISTENCY, COULD ITERATE ON PREV BUT NOT NEXT";
         end if;
      end loop;

   end Apply_To_Vertex_Ring;

   procedure Compute_Normals (Shape : in out Volume) is
      Normal : Point_Real := (0.0, 0.0, 0.0);

      V_Prev : Point_Real;
      V_Center : Point_Real;

      First_Forward : Boolean := True;
      First_Backward : Boolean := True;

      V_First : Point_Real;

      procedure Compute (Vert : Vertex; Forward : Boolean) is
         V_Cur : Point_Real;
      begin
         if Forward and then First_Forward then
            First_Forward := False;
            V_First := Vert.Point - V_Center;
            V_Prev := V_First;
         else
            if not Forward and then First_Backward then
               if First_Forward then
                  First_Forward := False;
                  V_First := Vert.Point - V_Center;
               end if;

               First_Backward := False;
               V_Prev := V_First;
            end if;

            V_Cur := Vert.Point - V_Center;

            if Forward then
               Normal := @ + Cross (V_Cur, V_Prev);
            else
               Normal := @ + Cross (V_Prev, V_Cur);
            end if;

            V_Prev := V_Cur;
         end if;
      end Compute;

   begin
      for V of Shape.Vertices loop
         Normal := (0.0, 0.0, 0.0);
         First_Forward := True;
         First_Backward := True;
         V_Center := V.Point;

         Apply_To_Vertex_Ring (Shape, Shape.Halfedges (V.Halfedge), Compute'Unrestricted_Access);

         V.Normal := Normalize (Normal);
      end loop;
   end Compute_Normals;

   function Get_Normal (Shape : Volume; Index : Volume_Index) return Point_Real is
   begin
      return Element (Shape.Vertices, Index).Normal;
   end Get_Normal;

end Volumes;
