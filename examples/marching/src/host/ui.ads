with Geometry; use Geometry;

package UI is

   procedure Initialize;

   procedure Finalize;

   procedure Draw (Verts : Vertex_Array; Tris : Triangle_Array; Running : out Boolean);

end UI;
