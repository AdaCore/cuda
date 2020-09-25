pragma Ada_2012;
package body CUDA.Device_Atomic_Functions is

   ----------------
   -- Atomic_Add --
   ----------------

   function Atomic_Add (Address : access int; Value : int) return int is
   begin
      return raise Program_Error with "Unimplemented function Atomic_Add";
   end Atomic_Add;

end CUDA.Device_Atomic_Functions;
