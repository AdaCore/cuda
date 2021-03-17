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

end CUDA.Runtime_Api;
