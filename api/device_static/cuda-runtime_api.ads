with Interfaces.C;
with CUDA.Vector_Types;

package CUDA.Runtime_Api is

   function Grid_Dim return CUDA.Vector_Types.Dim3 with
      Inline;
   function Block_Idx return CUDA.Vector_Types.Uint3 with
      Inline;
   function Block_Dim return CUDA.Vector_Types.Dim3 with
      Inline;
   function Thread_Idx return CUDA.Vector_Types.Uint3 with
      Inline;
   function Warp_Size return Interfaces.C.int with
      Inline;
   procedure Sync_Threads with Inline;
   --  bind CUDA procedure __syncthreads()
   --  which is a shorthand for LLVM intrinsic
   --    declare void @llvm.nvvm.barrier0()
   --  https://www.llvm.org/docs/NVPTXUsage.html#llvm-nvvm-barrier0

end CUDA.Runtime_Api;
