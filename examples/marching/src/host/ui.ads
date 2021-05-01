with Geometry; use Geometry;
with Ada.Numerics.Elementary_Functions;
use Ada.Numerics.Elementary_Functions;

package UI is

   procedure Initialize;

   procedure Finalize;

   procedure Draw (Verts : Vertex_Array; Tris : Triangle_Array; Running : out Boolean);

   function Length (R : Point_Real) return Float is
     (Sqrt (R.X ** 2 + R.Y ** 2 + R.Z ** 2));

   function Normalize (R : Point_Real) return Point_Real is
     (declare
      L : constant Float := Length (R);
      begin
        (if L = 0.0 then
             (0.0, 0.0, 0.0)
         else
            R / L));

end UI;
