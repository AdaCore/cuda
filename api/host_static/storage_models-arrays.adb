with Interfaces.C; use Interfaces.C;

package body Storage_Models.Arrays is

   function Length (First, Last : Index_Typ) return Natural is
      (Index_Typ'Pos (Last) - Index_Typ'Pos (First) + 1);

   --------------
   -- Allocate --
   --------------

   function Allocate (First, Last : Index_Typ) return Foreign_Array_Access is
      Ret : Foreign_Array_Access;
      Bounds : aliased Array_Typ_Bounds := (First, Last);
   begin
      Ret.Data := Allocate (Length (First, Last) * Typ'Size / 8);
      Ret.Bounds := Allocate (Array_Typ_Bounds'Size / 8);

      Copy_To_Foreign
        (Dst   => Ret.Bounds,
         Src   => Bounds'Address,
         Bytes => Array_Typ_Bounds'Size / 8);

      return Ret;
   end Allocate;

   -----------------------
   -- Allocate_And_Init --
   -----------------------

   function Allocate_And_Init (Src : Array_Typ) return Foreign_Array_Access is
      Ret : Foreign_Array_Access := Allocate (Src'First, Src'Last);
   begin
      Assign (Ret, Src);

      return Ret;
   end Allocate_And_Init;

   ----------
   -- Assign --
   ---------

   procedure Assign (Dst : Foreign_Array_Access; Src : Array_Typ) is
      Dst_Bounds : Array_Typ_Bounds := Bounds (Dst);
   begin
      Assign (Dst, Dst_Bounds.First, Dst_Bounds.Last, Src);
   end Assign;

   ----------
   -- Assign --
   ----------

   procedure Assign (Dst : Foreign_Array_Access; First, Last : Index_Typ; Src : Array_Typ) is
      Dst_Length : Natural := Length (First, Last);
   begin
      if Dst_Length /= Src'Length then
         raise Constraint_Error;
      end if;

      Copy_To_Foreign
        (Dst   => Dst.Data,
         Src   => Src'Address,
         Bytes => Dst_Length * Typ'Size / 8);
   end Assign;

   ------------
   -- Assign --
   ------------

   procedure Assign (Dst : in out Array_Typ; Src : Foreign_Array_Access) is
      Src_Bounds : Array_Typ_Bounds := Bounds (Src);
   begin
      Assign (Dst, Src, Src_Bounds.First, Src_Bounds.Last);
   end Assign;

   ------------
   -- Assign --
   ------------

   procedure Assign (Dst : in out Array_Typ; Src : Foreign_Array_Access; First, Last : Index_Typ) is
       Src_Length : Natural := Length (First, Last);
   begin
      if Src_Length /= Dst'Length then
         raise Constraint_Error;
      end if;

      Copy_To_Native
        (Dst   => Dst'Address,
         Src   => Src.Data,
         Bytes => Src_Length * Typ'Size / 8);
   end Assign;

   ------------
   -- Device --
   ------------

   function Uncheck_Convert (Src : Foreign_Array_Access) return Array_Access is
      Ret : Array_Access with Address => Src'Address, Import;
   begin
      return Ret;
   end Uncheck_Convert;

   ------------
   -- Bounds --
   ------------

   function Bounds (Src : Foreign_Array_Access) return Array_Typ_Bounds is
       Bounds : aliased Array_Typ_Bounds;
   begin
      Copy_To_Native
        (Dst   => Bounds'Address,
         Src   => Src.Bounds,
         Bytes => Array_Typ_Bounds'Size / 8);

      return Bounds;
   end Bounds;

   procedure Deallocate (Src : in out Foreign_Array_Access) is
   begin
      Deallocate (Src.Data);
      Deallocate (Src.Bounds);
   end Deallocate;

end Storage_Models.Arrays;
