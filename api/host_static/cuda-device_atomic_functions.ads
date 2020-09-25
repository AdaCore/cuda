with Interfaces.C; use Interfaces.C;

package CUDA.Device_Atomic_Functions is

   function Atomic_Add (Address : access int; Value : int) return int;

end CUDA.Device_Atomic_Functions;
