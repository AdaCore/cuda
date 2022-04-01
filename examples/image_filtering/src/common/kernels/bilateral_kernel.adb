With System;

With Cuda.Runtime_Api;
Use Cuda.Runtime_Api; -- Block_Dim, Block_Idx, Thread_Idx
 
With Interfaces.C;
Use Interfaces.C;     -- Operators For Block_Dim, Block_Idx, Thread_Idx

With Graphic;

Package Body Bilateral_Kernel Is

    Package G Renames Graphic; 

    Procedure Bilateral (Img_Addr          : System.Address; 
                         Filtered_Img_Addr : System.Address;
                         Width             : Integer;
                         Height            : Integer;
                         Spatial_Stdev     : Float;
                         Color_Dist_Stdev  : Float;
                         I                 : Integer;
                         J                 : Integer) Is

        Kernel_Size         : Integer := Integer (2.0 * Spatial_Stdev * 3.0);
        Half_Size           : Natural := (Kernel_Size - 1) / 2;

        Spatial_Gaussian    : Float := 0.0;
        Color_Dist_Gaussian : Float := 0.0;
        Sg_Cdg              : Float := 0.0;
        Sum_Sg_Cdg          : Float := 0.0;
        Rgb_Dist            : Float := 0.0;
        Filtered_Rgb        : G.Rgb := (0.0, 0.0, 0.0);

        Img          : G.Image (1 .. Width, 1 .. Height) With Address => Img_Addr;
        Filtered_Img : G.Image (1 .. Width, 1 .. Height) With Address => Filtered_Img_Addr;

        Function Exponential (N : Integer; X : Float) Return Float Is
            Sum : Float := 1.0;
        Begin
            For I In Reverse 1 .. N Loop
                Sum := 1.0 + X * Sum / Float(I);
            End Loop;
            Return Sum;
        End;

        Function Sqrt(X : Float; T : Float) Return Float Is
            Y : Float := 1.0;
        Begin
            While Abs (X/Y - Y) > T Loop
                Y := (Y + X / Y) / 2.0;
            End Loop;
            Return Y;
        End;

        Function Compute_Spatial_Gaussian (M : Float; N : Float) Return Float Is
            Spatial_Variance : Float := Spatial_Stdev * Spatial_Stdev;
            Two_Pi_Variance  : Float := 2.0*3.1416*Spatial_Variance;
            Exp              : Float := Exponential (10, -0.5 * ((M*M + N*N)/Spatial_Variance));
        Begin
            Return (1.0 / (Two_Pi_Variance)) * Exp;
        End;

        Function Compute_Color_Dist_Gaussian (K : Float) Return Float Is
            Color_Dist_Variance : Float := Color_Dist_Stdev * Color_Dist_Stdev;
        Begin
            Return (1.0 / (Sqrt(2.0*3.1416, 0.001)*Color_Dist_Stdev)) * Exponential (10, -0.5 * ((K*K)/Color_Dist_Variance));
        End;

        -- Compute Kernel Bounds
        Xb : Integer := I - Half_Size;
        Xe : Integer := I + Half_Size;
        Yb : Integer := J - Half_Size;
        Ye : Integer := J + Half_Size;

        Test : Float := 0.0;

    Begin
        For X In Xb .. Xe Loop
            For Y In Yb .. Ye Loop
                If X >= 1 And X <= Width And Y >= 1 And Y <= Height Then
                    -- Compute Color Distance
                    Rgb_Dist := Sqrt ((Img (I, J).R - Img (X, Y).R) * (Img (I, J).R - Img (X, Y).R) +
                                      (Img (I, J).G - Img (X, Y).G) * (Img (I, J).G - Img (X, Y).G) +
                                      (Img (I,J ).B - Img (X, Y).B) * (Img (I, J).B - Img (X, Y).B), 0.001);

                    -- Compute Gaussians
                    Spatial_Gaussian := Compute_Spatial_Gaussian (Float(I - X), Float(J - Y));
                    Color_Dist_Gaussian := Compute_Color_Dist_Gaussian (Rgb_Dist);

                    -- Multiply Gaussians
                    Sg_Cdg := Spatial_Gaussian*Color_Dist_Gaussian;

                    -- Accumulate
                    Filtered_Rgb.R := Filtered_Rgb.R + Sg_Cdg * Img (X,Y).R;
                    Filtered_Rgb.G := Filtered_Rgb.G + Sg_Cdg * Img (X,Y).G;
                    Filtered_Rgb.B := Filtered_Rgb.B + Sg_Cdg * Img (X,Y).B;

                    -- Track To Normalize Intensity
                    Sum_Sg_Cdg := Sum_Sg_Cdg + Sg_Cdg;
                End If;
            End Loop;
        End Loop;

        -- Normalize Intensity
        Filtered_Img (I,J).R := Filtered_Rgb.R / Sum_Sg_Cdg;
        Filtered_Img (I,J).G := Filtered_Rgb.G / Sum_Sg_Cdg;
        Filtered_Img (I,J).B := Filtered_Rgb.B / Sum_Sg_Cdg;
    End;

    Procedure Bilateral_Cuda (Device_Img          : System.Address; 
                              Device_Filtered_Img : System.Address;
                              Width               : Integer;
                              Height              : Integer;
                              Spatial_Stdev       : Float;
                              Color_Dist_Stdev    : Float) Is
        I : Integer := Integer (Block_Idx.X);
        J : Integer := Integer (Block_Idx.Y);
    Begin
        Bilateral (Device_Img, Device_Filtered_Img, Width, Height, Spatial_Stdev, Color_Dist_Stdev, I, J);
    End;

End Bilateral_Kernel;
