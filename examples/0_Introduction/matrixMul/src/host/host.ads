with CUDA.Vector_Types;         use CUDA.Vector_Types;

package Host is
    function Matrix_Multiply
        (Dims_A, Dims_B : Dim3; Measure_Performance : Boolean := False) return Integer;

end Host;
