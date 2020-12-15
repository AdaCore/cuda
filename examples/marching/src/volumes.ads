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

with Geometry;               use Geometry;
with Ada.Containers.Vectors; use Ada.Containers;

package Volumes is

   type Volume is private;

   subtype Volume_Index is Integer range -1 .. Integer'Last;
   type Volume_Indicies is array (1 .. 3) of Volume_Index;

   procedure Clear (Shape : in out Volume);

   function Create_Vertex
     (Shape : in out Volume;
      Point :        Point_Real) return Volume_Index;
   procedure Create_Face
     (Shape    : in out Volume;
      Indicies :        Volume_Indicies);

   function Get_Vertices
     (Shape : Volume;
      Index : Volume_Index) return Volume_Indicies;
   function Get_Vertex
     (Shape : Volume;
      Index : Volume_Index) return Point_Real;

   function First_Face_Index       (Shape : Volume) return Volume_Index;
   function Last_Face_Index        (Shape : Volume) return Volume_Index;
   function First_Vertex_Index     (Shape : Volume) return Volume_Index;
   function Last_Vertex_Index      (Shape : Volume) return Volume_Index;
   function Missing_Opposite_Count (Shape : Volume) return Integer;

   procedure Compute_Normals (Shape : in out Volume);
   function Get_Normal (Shape : Volume; Index : Volume_Index) return Point_Real;

private

   type Halfedge is record
      Vertex   : Volume_Index := -1;
      Face     : Volume_Index := -1; -- Adjacent face
      Next     : Volume_Index := -1; -- Next halfedge of the boundary
      Prev     : Volume_Index := -1; -- Previous halfedge of the boundary
      Opposite : Volume_Index := -1; -- The opposite (or inverse) halfedge
   end record;

   type Vertex is record
      Point    : Point_Real;
      Normal   : Point_Real;
      Halfedge : Volume_Index := -1; -- One of the halfedges thats connected
   end record;

   package Vertex_Vectors
     is new Ada.Containers.Vectors (Natural, Vertex);
   use Vertex_Vectors;

   package Halfedge_Index_Vectors
     is new Ada.Containers.Vectors (Natural, Volume_Index);
   use Halfedge_Index_Vectors;

   package Halfedge_Vectors
     is new Ada.Containers.Vectors (Natural, Halfedge);
   use Halfedge_Vectors;

   package Halfedge_Index_Vectors_Vectors
     is new Ada.Containers.Vectors (Natural, Halfedge_Index_Vectors.Vector);
   use Halfedge_Index_Vectors_Vectors;

   type Volume is record
      Halfedges         : Halfedge_Vectors.Vector;
      Faces             : Halfedge_Index_Vectors.Vector;
      Vertices          : Vertex_Vectors.Vector;
      Missing_Opposites : Halfedge_Index_Vectors_Vectors.Vector;
   end record;

   function First_Face_Index (Shape : Volume) return Volume_Index is
     (Volume_Index (Shape.Faces.First_Index));

   function Last_Face_Index (Shape : Volume) return Volume_Index is
     (Volume_Index (Shape.Faces.Last_Index));

   function First_Vertex_Index (Shape : Volume) return Volume_Index is
     (Volume_Index (Shape.Vertices.First_Index));

   function Last_Vertex_Index (Shape : Volume) return Volume_Index is
     (Volume_Index (Shape.Vertices.Last_Index));

   function Missing_Opposite_Count (Shape : Volume) return Integer is
     (Integer (Shape.Missing_Opposites.Length));

   function Get_Vertex
     (Shape : Volume;
      Index : Volume_Index) return Point_Real
   is (Shape.Vertices (Index).Point);

   function Get_Vertices
     (Shape : Volume;
      Index : Volume_Index) return Volume_Indicies
   is ((Shape.Halfedges (Shape.Faces (Index)).Vertex,
        Shape.Halfedges (Shape.Halfedges (Shape.Faces (Index)).Next).Vertex,
        Shape.Halfedges (Shape.Halfedges (Shape.Faces (Index)).Prev).Vertex));
end Volumes;





















