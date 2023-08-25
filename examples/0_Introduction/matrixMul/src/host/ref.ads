with Kernel;       use Kernel;
with Interfaces.C;     use Interfaces.C;

package Ref is
    procedure Matrix_Mul_Iter
     (C : out Float_Array;
      A : Float_Array;
      B : Float_Array;
      A_Width : unsigned;
      B_Width : unsigned);
    --  iterative version of matrix mutiplication
    procedure Test_Matrix_Mul_Iter;
end Ref;