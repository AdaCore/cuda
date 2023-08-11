with CUDA.Runtime_Api; use CUDA.Runtime_Api; -- Block_Dim, Block_IDx, Thread_IDx

package body Kernel is

   subtype Buffer_Range is unsigned range 0..Block_Size;
   type Buffer_Type is array (Buffer_Range, Buffer_Range) of Float;

   --      // Declaration of the shared memory array As used to
   --      // store the sub-matrix of A
   --      __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
   As : Buffer_Type;

   --      // Declaration of the shared memory array Bs used to
   --      // store the sub-matrix of B
   --      __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];
   Bs : Buffer_Type;

   procedure Matrix_Mul_CUDA
     (C : Array_Device_Access;
      A : Array_Device_Access;
      B : Array_Device_Access;
      A_Width : unsigned;
      B_Width : unsigned)
   is
      --    // Block index
      --    int bx = blockIdx.x;
      --    int by = blockIdx.y;
      Bl_x : unsigned := Block_IDx.X;
      Bl_y : unsigned := Block_IDx.Y;

      --    // Thread index
      --    int tx = threadIdx.x;
      --    int ty = threadIdx.y;
      T_x : unsigned :=  Thread_IDx.X;
      T_y : unsigned :=  Thread_IDx.Y;

      --    // Index of the first sub-matrix of A processed by the block
      --    int aBegin = wA * BLOCK_SIZE * by;
      A_Begin : unsigned := A_Width * Block_Size * Bl_y;

      --    // Index of the last sub-matrix of A processed by the block
      --    int aEnd   = aBegin + wA - 1;
      A_End : unsigned := A_Begin + A_Width - 1;

      --    // Step size used to iterate through the sub-matrices of A
      --    int aStep  = BLOCK_SIZE;
      A_Step : unsigned := Block_Size;

      --    // Index of the first sub-matrix of B processed by the block
      --    int bBegin = BLOCK_SIZE * bx;
      B_Begin : unsigned := Block_Size * Bl_x;

      --    // Step size used to iterate through the sub-matrices of B
      --    int bStep  = BLOCK_SIZE * wB;
      B_Step : unsigned := Block_Size * B_Width;

      --    // Csub is used to store the element of the block sub-matrix
      --    // that is computed by the thread
      --    float Csub = 0;
      C_Sub : Float := 0.0;

      A_Idx : unsigned := A_Begin;
      B_Idx : unsigned := B_Begin;
   begin
      --    // Loop over all the sub-matrices of A and B
      --    // required to compute the block sub-matrix
      --    for (int a = aBegin, b = bBegin;
      --         a <= aEnd;
      --         a += aStep, b += bStep) {

      while A_Idx <= A_End loop

         --      // Load the matrices from device memory
         --      // to shared memory; each thread loads
         --      // one element of each matrix
         --      As[ty][tx] = A[a + wA * ty + tx];
         --      Bs[ty][tx] = B[b + wB * ty + tx];
         As (T_y, T_x) := A (A_Idx + A_Width * T_y + T_x + 1);
         Bs (T_y, T_x) := B (B_Idx + B_Width * T_y + T_x + 1);

         --      // Synchronize to make sure the matrices are loaded
         --      __syncthreads();
         Sync_Threads;

         --      // Multiply the two matrices together;
         --      // each thread computes one element
         --      // of the block sub-matrix
         --  #pragma unroll

         --      for (int k = 0; k < BLOCK_SIZE; ++k) {
         for K in Buffer_Range loop
            --        Csub += As[ty][k] * Bs[k][tx];
            C_Sub := C_Sub + As (T_y, K) * Bs (K, T_x);
         --      }
         end loop;

         A_Idx := A_Idx + A_Step;
         B_Idx := B_Idx + B_Step;

         --      // Synchronize to make sure that the preceding
         --      // computation is done before loading two new
         --      // sub-matrices of A and B in the next iteration
         --      __syncthreads();
         Sync_Threads;
      --    }
      end loop;

      --    // Write the block sub-matrix to device memory;
      --    // each thread writes one element
      declare
         --    int c = wB * BLOCK_SIZE * by + BLOCK_SIZE * bx;
         C_Begin : unsigned := B_Width * Block_Size * Bl_y + Block_Size * Bl_x;
      begin
         --    C[c + wB * ty + tx] = Csub;
         C (C_Begin + B_Width * T_y + T_x + 1) := C_Sub;
      end;
   end Matrix_Mul_CUDA;

end Kernel;
