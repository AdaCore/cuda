package body CUDA.Internal is

   procedure Launch_Kernel
     (Func       : System.Address;
      Grid_Dim   : Dim3;
      Block_Dim  : Dim3;
      Args       : System.Address;
      Shared_Mem : Interfaces.C.Unsigned_Long;
      Stream     : CUDA.Driver_Types.Stream_T)
   is
      use Interfaces.C;

      function Internal
        (Func       : System.Address;
         Grid_Dim   : Dim3;
         Block_Dim  : Dim3;
         Args       : System.Address;
         Shared_Mem : Interfaces.C.Unsigned_Long;
         Stream     : CUDA.Driver_Types.Stream_T) return Interfaces.C.int
        with Import => True,
          Convention => C,
          External_Name => "cudaLaunchKernel";

      R : Interfaces.C.int;
   begin
      R := Internal (Func, Grid_Dim, Block_Dim, Args, Shared_Mem, Stream);

      if R /= 0 then
         raise Program_Error with "cudaLaunchKernel error:" & R'Img;
      end if;
   end Launch_Kernel;

end CUDA.Internal;
