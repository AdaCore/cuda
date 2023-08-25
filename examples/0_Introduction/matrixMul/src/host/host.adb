--  with System;
with Interfaces.C;               use Interfaces.C;

with CUDA.Driver_Types;          use CUDA.Driver_Types;
with CUDA.Runtime_Api;           use CUDA.Runtime_Api;
with CUDA.Stddef;
with udriver_types_h;
--  with CUDA.Storage_Models;        use CUDA.Storage_Models;
with CUDA.Vector_Types;          use CUDA.Vector_Types;

with Kernel; use Kernel;
with Ref;    use Ref;

with Ada.Unchecked_Deallocation;
with Ada.Text_IO;                use Ada.Text_IO;

package body Host is
    --  /**
    --   * Run a simple test of matrix multiplication using CUDA
    --   */
    --  int MatrixMultiply(int argc, char **argv,
    --                     int block_size, const dim3 &dimsA,
    --                     const dim3 &dimsB) {
    function Matrix_Multiply
        (Dims_A, Dims_B : Dim3; Measure_Performance : Boolean := False) return Integer
    is
        type Array_Host_Access is access all Float_Array;

        procedure Free is new
            Ada.Unchecked_Deallocation (Float_Array, Array_Host_Access);

        procedure Free is new Ada.Unchecked_Deallocation
            (Float_Array, Array_Device_Access);

        --    // Allocate host memory for matrices A and B
        --    unsigned int size_A = dimsA.x * dimsA.y;
        Size_A : unsigned := Dims_A.x * Dims_A.y;

        --    unsigned int mem_size_A = sizeof(float) * size_A;
        --  Mem_Size_A : CUDA.Stddef.Size_T := CUDA.Stddef.Size_T(Interfaces.C.c_float'size * Size_A / 8);
        --  TODO: not sure about "/ 8". This was done in vectorAdd

        --    float *h_A;
        H_A : Array_Host_Access;

        --    unsigned int size_B = dimsB.x * dimsB.y;
        Size_B : unsigned := Dims_B.x * Dims_B.y;

        --    unsigned int mem_size_B = sizeof(float) * size_B;
        --  Mem_Size_B : CUDA.Stddef.Size_T := CUDA.Stddef.Size_T(Interfaces.C.c_float'size * Size_B / 8);

        --    float *h_B;
        H_B : Array_Host_Access;

        --    cudaStream_t stream;
        Stream : CUDA.Driver_Types.Stream_T;

        --    cudaEvent_t start, stop;
        Start_Event, Stop_Event : CUDA.Driver_Types.Event_T;

        --    // Initialize host memory
        --    const float valB = 0.01f;
        Val_B : constant Float := 0.01;

        --    // Allocate device memory
        --    float *d_A, *d_B, *d_C;
        D_A, D_B, D_C : Array_Device_Access;

        --    // Allocate host matrix C
        --    dim3 dimsC(dimsB.x, dimsA.y, 1);
        Dims_C : Dim3 := (Dims_B.x, Dims_A.y, 1);

        --    unsigned int mem_size_C = dimsC.x * dimsC.y * sizeof(float);
        Size_C : unsigned := Dims_C.x * Dims_C.y;
        --  Mem_Size_C : CUDA.Stddef.Size_T := CUDA.Stddef.Size_T(Interfaces.C.c_float'size * Dims_C.x * Dims_C.y / 8);

        --    float *h_C;
        H_C : Array_Host_Access;

        --    // Setup execution parameters
        --    dim3 threads(block_size, block_size);
        --    dim3 grid(dimsB.x / threads.x, dimsA.y / threads.y);
        Threads_Per_Block : Dim3 := (Block_Size, Block_Size, 1);
        Blocks_Per_Grid : Dim3 :=
            (Dims_B.X / Threads_Per_Block.X, Dims_A.y / Threads_Per_Block.y, 1);

        --    bool correct = true;
        Res : Integer := 0;

        --  alternative output array for computing without cuda
        C_Iter : Float_Array (1..Size_C);
    begin
        --    checkCudaErrors(cudaMallocHost(&h_A, mem_size_A));
        --    checkCudaErrors(cudaMallocHost(&h_B, mem_size_B));
        --    checkCudaErrors(cudaMallocHost(&h_C, mem_size_C));
        H_A := new Float_Array (1..Size_A);
        H_B := new Float_Array (1..Size_B);
        H_C := new Float_Array (1..Size_C);
        --    ConstantInit(h_A, size_A, 1.0f);
        --    ConstantInit(h_B, size_B, valB);
        H_A.all := (others => 1.0);
        H_B.all := (others => Val_B);

        --    if (h_C == NULL) {
        --      fprintf(stderr, "Failed to allocate host matrix C!\n");
        --      exit(EXIT_FAILURE);
        --    }
        --  TODO: shall this be done explicitly or we get already
        --  an appropriate error from "new" above?

        --    checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_A), mem_size_A));
        --    checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_B), mem_size_B));
        --    checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_C), mem_size_C));
        --    // copy host memory to device
        --    checkCudaErrors(
        --        cudaMemcpyAsync(d_A, h_A, mem_size_A, cudaMemcpyHostToDevice, stream));
        --    checkCudaErrors(
        --        cudaMemcpyAsync(d_B, h_B, mem_size_B, cudaMemcpyHostToDevice, stream));
        D_A := new Float_Array'(H_A.all);
        D_B := new Float_Array'(H_B.all);
        D_C := new Float_Array (H_C.all'Range);

        --    // Allocate CUDA events that we'll use for timing
        --    checkCudaErrors(cudaEventCreate(&start));
        --    checkCudaErrors(cudaEventCreate(&stop));
        Start_Event := Event_Create;
        Stop_Event := Event_Create;

        --    checkCudaErrors(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        Stream := Stream_Create_With_Flags (udriver_types_h.cudaStreamNonBlocking);

        --    // Create and start timer
        --    printf("Computing result using CUDA Kernel...\n");
        Put_Line ("Computing result using CUDA Kernel...");

        --    // Performs warmup operation using matrixMul CUDA kernel
        --    if (block_size == 16) {
        --      MatrixMulCUDA<16>
        --          <<<grid, threads, 0, stream>>>(d_C, d_A, d_B, dimsA.x, dimsB.x);
        --    } else {
        --      MatrixMulCUDA<32>
        --          <<<grid, threads, 0, stream>>>(d_C, d_A, d_B, dimsA.x, dimsB.x);
        --    }
        pragma CUDA_Execute
            (Matrix_Mul_CUDA (D_C, D_A, D_B,
                Dims_A.X, Dims_B.X),
                Blocks_Per_Grid,
                Threads_Per_Block,
                0,
                Stream);

        --    printf("done\n");
        Put_Line ("done");

        --    checkCudaErrors(cudaStreamSynchronize(stream));
        Stream_Synchronize (Stream);

        if Measure_Performance then
            --    // Record the start event
            --    checkCudaErrors(cudaEventRecord(start, stream));
            Event_Record (Start_Event, Stream);

            --    // Execute the kernel N_Iter times
            declare
                --    int nIter = 300;
                N_Iter : constant Integer := 300;
                --    float msecTotal = 0.0f;
                MSec_Total : Float := 0.0;
                MSec_Per_Matrix_Mul : Float;
                Flops_Per_Matrix_Mul : Long_Float;
                Giga_Flops : Long_Float;
                Workgroup_Size : constant unsigned :=
                    Threads_Per_Block.X * Threads_Per_Block.Y;
            begin
                --    for (int j = 0; j < nIter; j++) {
                for J in 1 .. N_Iter loop
                    --      if (block_size == 16) {
                    --        MatrixMulCUDA<16>
                    --            <<<grid, threads, 0, stream>>>(d_C, d_A, d_B, dimsA.x, dimsB.x);
                    --      } else {
                    --        MatrixMulCUDA<32>
                    --            <<<grid, threads, 0, stream>>>(d_C, d_A, d_B, dimsA.x, dimsB.x);
                    --      }
                    pragma CUDA_Execute
                        (Matrix_Mul_CUDA (D_C, D_A, D_B,
                            Dims_A.X, Dims_B.X),
                            Blocks_Per_Grid,
                            Threads_Per_Block);
                --    }
                end loop;

                --    // Record the stop event
                --    checkCudaErrors(cudaEventRecord(stop, stream));
                Event_Record (Stop_Event, Stream);

                --    // Wait for the stop event to complete
                --    checkCudaErrors(cudaEventSynchronize(stop));
                Event_Synchronize (Stop_Event);

                --    checkCudaErrors(cudaEventElapsedTime(&msecTotal, start, stop));
                MSec_Total := Event_Elapsed_Time (Start_Event, Stop_Event);

                --    // Compute and print the performance
                --    float msecPerMatrixMul = msecTotal / nIter;
                MSec_Per_Matrix_Mul := MSec_Total / Float (N_Iter);

                --    double flopsPerMatrixMul = 2.0 * static_cast<double>(dimsA.x) *
                --                               static_cast<double>(dimsA.y) *
                --                               static_cast<double>(dimsB.x);
                Flops_Per_Matrix_Mul := 2.0 * Long_Float (Dims_A.X) +
                    Long_Float (Dims_A.Y) * Long_Float (Dims_B.X);
                --    double gigaFlops =
                --        (flopsPerMatrixMul * 1.0e-9f) / (msecPerMatrixMul / 1000.0f);
                Giga_Flops := (Flops_Per_Matrix_Mul * Long_Float (1.0e-9)) /
                    Long_Float (MSec_Per_Matrix_Mul / 1000.0);
                --    printf(
                --        "Performance= %.2f GFlop/s, Time= %.3f msec, Size= %.0f Ops,"
                --        " WorkgroupSize= %u threads/block\n",
                --        gigaFlops, msecPerMatrixMul, flopsPerMatrixMul, threads.x * threads.y);
                Put_Line (
                    "Performance=" & Giga_Flops'Img &
                    " GFlop/s, Time=" & MSec_Per_Matrix_Mul'Img &
                    " msec, Size=" & Flops_Per_Matrix_Mul'Img &
                    " Ops, WorkgroupSize=" & Workgroup_Size'Img &
                    " threads/block.");
            end;
        end if;

        --    // Copy result from device to host
        --    checkCudaErrors(
        --        cudaMemcpyAsync(h_C, d_C, mem_size_C, cudaMemcpyDeviceToHost, stream));
        H_C.all := D_C.all;

        --    checkCudaErrors(cudaStreamSynchronize(stream));
        Stream_Synchronize (Stream);

        Put_Line ("Computing result using Iteration...");
        Matrix_Mul_Iter(C_Iter, H_A.all, H_B.all, Dims_A.X, Dims_B.X);
        Put_Line ("done");

        --    printf("Checking computed result for correctness: ");
        Put_Line ("Checking computed result for correctness: ");

        declare
            Correct_Cnt : Natural := 0;
            Error_Cnt : Natural := 0;

            Correct_Cnt_I : Natural := 0;
            Error_Cnt_I : Natural := 0;

            --    // test relative error by the formula
            --    //     |<x, y>_cpu - <x,y>_gpu|/<|x|, |y|>  < eps
            --    double eps = 1.e-6;  // machine zero
            Eps : Float := 1.0e-6;
        begin

            --    for (int i = 0; i < static_cast<int>(dimsC.x * dimsC.y); i++) {
            for I in 1 .. Dims_C.x * Dims_C.y loop
                declare
                    Ref : Float := Float (Dims_A.x) * Val_B;
                    --      double abs_err = fabs(h_C[i] - (dimsA.x * valB));
                    Abs_Err : Float := Abs (H_C (I) - Ref);
                    Abs_Err_I : Float := Abs (C_Iter (I) - Ref);

                    --      double dot_length = dimsA.x;
                    Dot_Length : Float := FLoat (Dims_A.x);
                    --      double abs_val = fabs(h_C[i]);
                    Abs_Val : Float := Abs (H_C (I));
                    Abs_Val_I : Float := Abs (C_Iter (I));

                    --      double rel_err = abs_err / abs_val / dot_length;
                    Rel_Err : Float := Abs_Err / Abs_Val / Dot_Length;
                    Rel_Err_I : Float := Abs_Err_I / Abs_Val_I / Dot_Length;
                begin
                    --      if (rel_err > eps) {
                    --        printf("Error! Matrix[%05d]=%.8f, ref=%.8f error term is > %E\n",
                    --               i, h_C[i], dimsA.x * valB, eps);
                    --        correct = false;
                    --      }
                    if Rel_Err > Eps then
                        Put_Line ("Error! Matrix[" & I'Img & "]=" &
                                  H_C (I)'Img & ", ref=" & Ref'Img &
                                  " error term is > " & Eps'Img);
                        Error_Cnt := Error_Cnt + 1;
                    else
                        Correct_Cnt := Correct_Cnt + 1;
                    end if;
                    if Rel_Err_I > Eps then
                        Put_Line ("Error! Matrix[" & I'Img & "]=" &
                                  C_Iter (I)'Img & ", ref=" & Ref'Img &
                                  " error term is > " & Eps'Img);
                        Error_Cnt_I := Error_Cnt_I + 1;
                    else
                        Correct_Cnt_I := Correct_Cnt_I + 1;
                    end if;
                end;
            --    }
            end loop;

            --    printf("%s\n", correct ? "Result = PASS" : "Result = FAIL");
            Put_Line ("Correct cuda:" & Correct_Cnt'Img & ", errors:" & Error_Cnt'Img);
            Put_Line ("Correct iter:" & Correct_Cnt_I'Img & ", errors:" & Error_Cnt_I'Img);

            if Error_Cnt = 0 then
                Put_Line ("Result = PASS");
                Res := 0;
            else
                Put_Line ("Result = FAIL");
                Res := 1;
            end if;
        end;

        --    // Clean up memory
        --    checkCudaErrors(cudaFreeHost(h_A));
        --    checkCudaErrors(cudaFreeHost(h_B));
        --    checkCudaErrors(cudaFreeHost(h_C));
        Free (H_A);
        Free (H_B);
        Free (H_C);

        --    checkCudaErrors(cudaFree(d_A));
        --    checkCudaErrors(cudaFree(d_B));
        --    checkCudaErrors(cudaFree(d_C));
        Free (D_A);
        Free (D_B);
        Free (D_C);

        --    checkCudaErrors(cudaEventDestroy(start));
        --    checkCudaErrors(cudaEventDestroy(stop));
        Event_Destroy (Start_Event);
        Event_Destroy (Stop_Event);

        if Measure_Performance then
            --    printf(
            --        "\nNOTE: The CUDA Samples are not meant for performance "
            --        "measurements. Results may vary when GPU Boost is enabled.\n");
            Put_Line ("");
            Put_Line ("NOTE: The CUDA Samples are not meant for performance");
            Put_LIne ("measurements. Results may vary when GPU Boost is enabled.");
        end if;

        --    if (correct) {
        --      return EXIT_SUCCESS;
        --    } else {
        --      return EXIT_FAILURE;
        --    }
        return Res;
        --  }
    end Matrix_Multiply;

end Host;
