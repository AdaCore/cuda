With Ada.Text_IO; use Ada.Text_IO;

package body Ref is
    procedure Matrix_Mul_Iter
     (C : out Float_Array;
      A : Float_Array;
      B : Float_Array;
      A_Width : unsigned;
      B_Width : unsigned)
    is
        M : unsigned := A'Length / A_Width;
        N : unsigned renames A_Width;
        P : unsigned renames B_Width;
        A_Idx : unsigned;
        B_Idx : unsigned;
        C_Idx : unsigned;

        function Serialize_Idx (X, Y, Width : unsigned) return unsigned is
          ((Y - 1) * Width + X);
    begin
        for M_Idx in 1 .. M loop
            for P_Idx in 1 .. P loop
                C_Idx := Serialize_Idx (M_Idx, P_Idx, M);
                C (C_Idx) := 0.0;
                for N_Idx in 1 .. N loop
                    A_Idx := Serialize_Idx (N_Idx, M_Idx, N);
                    B_Idx := Serialize_Idx (P_Idx, N_Idx, P);
                    C (C_Idx) := C (C_Idx) +
                        A (A_Idx) * B (B_Idx);
                end loop;
            end loop;
        end loop;
    end Matrix_Mul_Iter;

    procedure Test_Matrix_Mul_Iter
    is
        A_Width : constant unsigned := 3;
        A_Height : constant unsigned := 1;
        B_Width : constant unsigned := 4;
        B_Height : constant unsigned := A_Width;
        A : Float_Array (1 .. A_Width * A_Height) := (3.0, 4.0, 2.0);
        B : Float_Array (1 .. B_Width * B_Height) :=
            (13.0, 9.0, 7.0, 15.0,
             8.0, 7.0, 4.0, 6.0,
             6.0, 4.0, 0.0, 3.0);
        C : Float_Array (1 .. A_Height * B_Width);
    begin
        Matrix_Mul_Iter (C, A, B, A_Width, B_Width);
        for I in C'Range loop
            Put_Line (C (I)'Img);
        end loop;
    end Test_Matrix_Mul_Iter;
end Ref;