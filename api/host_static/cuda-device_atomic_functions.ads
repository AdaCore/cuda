with Interfaces.C; use Interfaces.C;

package CUDA.Device_Atomic_Functions is

   function Atomic_Add
     (Address : access int; Value : int; Ordering : int := 0) return int with
      Convention => Intrinsic,
      Import,
      External_Name => "__atomic_fetch_add_4";

end CUDA.Device_Atomic_Functions;
