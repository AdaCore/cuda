with System;

with CUDA.Storage_Models; use CUDA.Storage_Models;

with Interfaces.C;     use Interfaces.C;

package Kernel is

   Block_Size : constant unsigned := 32;

   type Float_Array is array (unsigned range <>) of Float;

   type Array_Device_Access is access Float_Array
     with Designated_Storage_Model => CUDA.Storage_Models.Model;

--  /**
--   * Matrix multiplication (CUDA Kernel) on the device: C = A * B
--   * wA is A's width and wB is B's width
--   */
--  template <int BLOCK_SIZE> __global__ void MatrixMulCUDA(float *C, float *A,
--      float *B, int wA,
--      int wB) {
    procedure Matrix_Mul_CUDA
     (C : Array_Device_Access;
      A : Array_Device_Access;
      B : Array_Device_Access;
      A_Width : unsigned;
      B_Width : unsigned)
     with CUDA_Global;

end Kernel;
--