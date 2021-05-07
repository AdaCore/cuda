with Interfaces.C; use Interfaces.C;

package body Storage_Models.Arrays is

   function Length (First, Last : Index_Typ) return Natural is
     (Index_Typ'Pos (Last) - Index_Typ'Pos (First) + 1);

   function Compute_Offset (First, Last : Index_Typ) return Natural is
      (Index_Typ'Pos (Last) - Index_Typ'Pos (First));

   --------------
   -- Allocate --
   --------------

   function Allocate (First, Last : Index_Typ) return Array_Access is
      Ret : Array_Access;
      Bounds : Array_Typ'Bounds := (First, Last);
   begin
      Ret := Array_Typ'Fat_Pointer
        (Data   => Allocate (Length (First, Last) * Typ'Size / 8);
         Bounds => Allocate (Array_Typ_Bounds'Size / 8));

      Copy_To_Foreign
        (Dst     => Ret'Fat_Pointer.Bounds.all'Address,
         Src     => Bounds'Address,
         Bytes   => Array_Typ_Bounds'Size / 8,
         Options => Default_Copy_Options);

      return Ret;
   end Allocate;

   -----------------------
   -- Allocate_And_Init --
   -----------------------

   function Allocate_And_Init (Src : Array_Typ) return Array_Access is
      Ret : Foreign_Array_Access := Allocate (Src'First, Src'Last);
   begin
      Assign (Ret, Src, Default_Copy_Options);

      return Ret;
   end Allocate_And_Init;

   ----------
   -- Assign --
   ---------

   procedure Assign (Dst : Foreign_Array_Access; Src : Array_Typ; Options : Copy_Options := Default_Copy_Options) is
      Dst_Bounds : Array_Typ_Bounds := Bounds (Dst);
   begin
      Assign (Dst, Dst_Bounds.First, Dst_Bounds.Last, Src, Options);
   end Assign;

   ----------
   -- Assign --
   ----------

   procedure Assign (Dst : Foreign_Array_Access; First, Last : Index_Typ; Src : Array_Typ; Options : Copy_Options := Default_Copy_Options) is
      Dst_Length : Natural := Length (First, Last);
      Dst_Bounds : Array_Typ_Bounds := Bounds (Dst);
   begin
      if Dst_Length /= Src'Length then
         raise Constraint_Error;
      end if;

      Copy_To_Foreign
        (Dst     => Offset (Dst.Data, Compute_Offset (Dst_Bounds.First, First) * Typ'Size / 8),
         Src     => Src'Address,
         Bytes   => Dst_Length * Typ'Size / 8,
         Options => Options);
   end Assign;

   ----------
   -- Assign --
   ---------

   procedure Assign (Dst : Foreign_Array_Access; Src : Typ; Options : Copy_Options := Default_Copy_Options) is
      Dst_Bounds : Array_Typ_Bounds := Bounds (Dst);
   begin
      Assign (Dst, Dst_Bounds.First, Dst_Bounds.Last, Src, Options);
   end Assign;


   ----------
   -- Assign --
   ----------

   procedure Assign (Dst : Foreign_Array_Access; First, Last : Index_Typ; Src : Typ; Options : Copy_Options := Default_Copy_Options) is
      Dst_Length : Natural := Length (First, Last);
      Dst_Bounds : Array_Typ_Bounds := Bounds (Dst);
      Start_Offset : Natural := Compute_Offset (Dst_Bounds.First, First);

      Chunk : aliased array (1 .. 1024) of Typ := (others => Src);
      Remainder : Integer := 0;
   begin
      for D in 0 .. Dst_Length / 1024 loop
         Remainder := Dst_Length - D * 1024;

         if Remainder > 1024 then
            Remainder := 1024;
         end if;

         Copy_To_Foreign
           (Dst     => Offset (Dst.Data, (Start_Offset + D * 1024) * Typ'Size / 8),
            Src     => Chunk'Address,
            Bytes   => Remainder * Typ'Size / 8,
            Options => Options);
      end loop;
   end Assign;

   ------------
   -- Assign --
   ------------

   procedure Assign (Dst : in out Array_Typ; Src : Foreign_Array_Access; Options : Copy_Options := Default_Copy_Options) is
      Src_Bounds : Array_Typ_Bounds := Bounds (Src);
   begin
      Assign (Dst, Src, Src_Bounds.First, Src_Bounds.Last, Options);
   end Assign;

   ------------
   -- Assign --
   ------------

   procedure Assign (Dst : in out Array_Typ; Src : Foreign_Array_Access; First, Last : Index_Typ; Options : Copy_Options := Default_Copy_Options) is
      Src_Length : Natural := Length (First, Last);
      Src_Bounds : Array_Typ_Bounds := Bounds (Src);
   begin
      if Src_Length /= Dst'Length then
         raise Constraint_Error;
      end if;

      Copy_To_Native
        (Dst   => Dst'Address,
         Src   => Offset (Src.Data, Compute_Offset (Src_Bounds.First, First) * Typ'Size / 8),
         Bytes => Src_Length * Typ'Size / 8,
         Options => Options);
   end Assign;

   ------------
   -- Uncheck_Convert --
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
        (Dst     => Bounds'Address,
         Src     => Src.Bounds,
         Bytes   => Array_Typ_Bounds'Size / 8,
         Options => Default_Copy_Options);

      return Bounds;
   end Bounds;

   procedure Deallocate (Src : in out Foreign_Array_Access) is
   begin
      Deallocate (Src.Data);
      Deallocate (Src.Bounds);
   end Deallocate;

   function Allocate (Src : Foreign_Array_Access; First, Last : Index_Typ) return Foreign_Array_Slice_Access is
      Ret : Foreign_Array_Slice_Access;
      Src_Bounds : Array_Typ_Bounds := Bounds (Src);
      Bounds : aliased Array_Typ_Bounds := (First, Last);
   begin
      Ret.Data := Offset (Src.Data, Compute_Offset (Src_Bounds.First, First) * Typ'Size / 8);
      Ret.Bounds := Allocate (Array_Typ_Bounds'Size / 8);

      Copy_To_Foreign
        (Dst     => Ret.Bounds,
         Src     => Bounds'Address,
         Bytes   => Array_Typ_Bounds'Size / 8,
         Options => Default_Copy_Options);

      return Ret;
   end Allocate;

   procedure Deallocate (Src : in out Foreign_Array_Slice_Access) is
   begin
      Deallocate (Src.Bounds);
   end Deallocate;

end Storage_Models.Arrays;
