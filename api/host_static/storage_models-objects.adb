with Interfaces.C; use Interfaces.C;

package body Storage_Models.Objects is

   --------------
   -- Allocate --
   --------------

   function Allocate return Foreign_Access is
   begin
      return Foreign_Access (Allocate (Typ'Size / 8));
   end Allocate;

   -----------------------
   -- Allocate_And_Init --
   -----------------------

   function Allocate_And_Init (Src : Typ) return Foreign_Access is
      Ret : Foreign_Access := Allocate;
   begin
      Assign (Ret, Src);

      return Ret;
   end Allocate_And_Init;

   ------------
   -- Assign --
   ------------

   procedure Assign (Dst: Foreign_Access; Src : Typ; Options : Copy_Options := Default_Copy_Options) is
   begin
      Copy_To_Foreign
        (Dst     => Foreign_Address (Dst),
         Src     => Src'Address,
         Bytes   => Typ'Size / 8,
         Options => Options);
   end Assign;

   ------------
   -- Assign --
   ------------

   procedure Assign (Dst : in out Typ; Src : Foreign_Access; Options : Copy_Options := Default_Copy_Options) is
   begin
      Copy_To_Native
        (Dst     => Dst'Address,
         Src     => Foreign_Address (Src),
         Bytes   => Typ'Size / 8,
         Options => Options);
   end Assign;

   ------------
   -- Device --
   ------------

   function Uncheck_Convert (Src : Foreign_Access) return Typ_Access is
      Ret : Typ_Access with Address => Src'Address, Import;
   begin
      return Ret;
   end Uncheck_Convert;

   ----------------
   -- Deallocate --
   ----------------

   procedure Deallocate (Src : in out Foreign_Access) is
   begin
      Deallocate (Foreign_Address (Src));
   end Deallocate;

end Storage_Models.Objects;
