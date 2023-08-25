with Interfaces.C;      use Interfaces.C;

with GNAT.Command_Line; use GNAT.Command_Line;
with Ada.Text_IO;       use Ada.Text_IO;
with Ada.Strings;       use Ada.Strings;
with Ada.Strings.Fixed; use Ada.Strings.Fixed;

with CUDA.Vector_Types; use CUDA.Vector_Types;

with Host;              use Host;
with Kernel;            use Kernel;

function Main return Integer is

   --    int block_size = 32;
   --  MOVED TO KERNEL. In C implementation this is passed to kernel by
   --  template parameter. Bringing in generics didn't seem to add any
   --  any extra value here, so we use a global constant instead

   --    dim3 dimsA(5 * 2 * block_size, 5 * 2 * block_size, 1);
   --    dim3 dimsB(5 * 4 * block_size, 5 * 2 * block_size, 1);
   Dims_A : Dim3 := (5 * 2 * Block_Size, 5 * 2 * Block_Size, 1);
   Dims_B : Dim3 := (5 * 4 * Block_Size, 5 * 2 * Block_Size, 1);

   Measure_Performance : Boolean := False;
   Unknown_Argument : Boolean := False;

   Usage_Msg : constant String :=
      --  "Usage -device=n (n >= 0 for deviceID)" & ASCII.LF &
      "Usage [? | --help]" & ASCII.LF &
      "      [-wA=WidthA] [-hA=HeightA] (Width x Height of Matrix A)" & ASCII.LF &
      "      [-wB=WidthB] [-hB=HeightB] (Width x Height of Matrix B)" & ASCII.LF &
      "      [-p | --perf] print performance measurements." & ASCII.LF & ASCII.LF &
      "  Note: Outer matrix dimensions of A & B matrices must be equal.";
begin
   --    printf("[Matrix Multiply Using CUDA] - Starting...\n");
   Put_Line ("[Matrix Multiply Using CUDA] - Starting...");

   --    if (checkCmdLineFlag(argc, (const char **)argv, "help") ||
   --        checkCmdLineFlag(argc, (const char **)argv, "?")) {
   --      printf("Usage -device=n (n >= 0 for deviceID)\n");
   --      printf("      -wA=WidthA -hA=HeightA (Width x Height of Matrix A)\n");
   --      printf("      -wB=WidthB -hB=HeightB (Width x Height of Matrix B)\n");
   --      printf("  Note: Outer matrix dimensions of A & B matrices" \
   --             " must be equal.\n");

   --      exit(EXIT_SUCCESS);
   --    }

   --    // This will pick the best possible CUDA capable device, otherwise
   --    // override the device ID based on input provided at the command line
   --    int dev = findCudaDevice(argc, (const char **)argv);

   --    // width of Matrix A
   --    if (checkCmdLineFlag(argc, (const char **)argv, "wA")) {
   --      dimsA.x = getCmdLineArgumentInt(argc, (const char **)argv, "wA");
   --    }

   --    // height of Matrix A
   --    if (checkCmdLineFlag(argc, (const char **)argv, "hA")) {
   --      dimsA.y = getCmdLineArgumentInt(argc, (const char **)argv, "hA");
   --    }

   --    // width of Matrix B
   --    if (checkCmdLineFlag(argc, (const char **)argv, "wB")) {
   --      dimsB.x = getCmdLineArgumentInt(argc, (const char **)argv, "wB");
   --    }

   --    // height of Matrix B
   --    if (checkCmdLineFlag(argc, (const char **)argv, "hB")) {
   --      dimsB.y = getCmdLineArgumentInt(argc, (const char **)argv, "hB");
   --    }
   loop
      case Getopt ("? -help wA= hA= wB= hB= p -perf") is
         when '-' | '?' =>
            Put_Line (Usage_Msg);
            return 0;
         when 'w' =>
            if Full_Switch = "wA" then
               Dims_A.X := unsigned'Value (Parameter);
            elsif Full_Switch = "wB" then
               Dims_B.X := unsigned'Value (Parameter);
            else
               Unknown_Argument := True;
            end if;
         when 'h' =>
            if Full_Switch = "hA" then
               Dims_A.Y := unsigned'Value (Parameter);
            elsif Full_Switch = "hB" then
               Dims_B.Y := unsigned'Value (Parameter);
            else
               Unknown_Argument := True;
            end if;
         when 'p' =>
            if Full_Switch = "p" or else Full_Switch = "-perf" then
               Measure_Performance := True;
            else
               Unknown_Argument := True;
            end if;
         when others =>
            exit;
      end case;

      if Unknown_Argument then
            Put_Line ("ERROR: unknown argument " & Full_Switch);
            Put_Line (Usage_Msg);
            return 1;
      end if;
   end loop;

   --    if (dimsA.x != dimsB.y) {s
   --      printf("Error: outer matrix dimensions must be equal. (%d != %d)\n",
   --             dimsA.x, dimsB.y);
   --      exit(EXIT_FAILURE);
   --    }
   if Dims_A.x /= Dims_B.y then
      raise Program_Error with
         ("Error: outer matrix dimensions must be equal. (" &
            Dims_A.x'Img & " !=" & Dims_B.y'Img & ")");
   end if;

   --    printf("MatrixA(%d,%d), MatrixB(%d,%d)\n", dimsA.x, dimsA.y,
   --           dimsB.x, dimsB.y);
   Put_Line
      ("MatrixA (" & Trim (Dims_A.x'Img, Left) & "," & Dims_A.y'Img &
       "), MatrixB (" & Trim (Dims_B.x'Img, Left) & "," & Dims_B.y'Img & ")");

   return Res : Integer do
      --    checkCudaErrors(cudaProfilerStart());

      --    int matrix_result = MatrixMultiply(argc, argv, block_size, dimsA, dimsB);
      Res := Matrix_Multiply (Dims_A, Dims_B, Measure_Performance);
      --  the result is already printed in Matrix multiply, no need to handle the value here

      --    checkCudaErrors(cudaProfilerStop());

      --    exit(matrix_result);
   end return;
end Main;
