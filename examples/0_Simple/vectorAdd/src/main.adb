with System;
with Interfaces.C;              use Interfaces.C;

with Ada.Numerics.Float_Random; use Ada.Numerics.Float_Random;
with Ada.Text_IO;               use Ada.Text_IO;

with CUDA.Driver_Types;         use CUDA.Driver_Types;
with CUDA.Runtime_Api;          use CUDA.Runtime_Api;
with CUDA.Stddef;

with Kernel; use Kernel;

with Ada.Unchecked_Deallocation;

procedure Main is

   Num_Elements : Integer := 4096;

   H_A, H_B, H_C : Access_Host_Float_Array;
   D_A, D_B, D_C : System.Address;
   Array_Size : CUDA.Stddef.Size_T := CUDA.Stddef.Size_T(Interfaces.C.c_float'size * Num_Elements / 8);

   Threads_Per_Block : Integer := 256;
   Blocks_Per_Grid : Integer :=
     (Num_Elements + Threads_Per_Block - 1) / Threads_Per_Block;

   Gen : Generator;
   Err : Error_T;

   procedure Free is new
     Ada.Unchecked_Deallocation (Float_Array, Access_Host_Float_Array);

begin
   Put_Line ("[Vector addition of " & Num_Elements'Img & " elements]");

   H_A := new Float_Array (1 .. Num_Elements);
   H_B := new Float_Array (1 .. Num_Elements);
   H_C := new Float_Array (1 .. Num_Elements);

   if H_A = null or else H_B = null or else H_C = null then
      Put_Line("Failed to allocate host vectors!");

      return;
   end if;

   H_A.all := (others => Float (Random (Gen)));
   H_B.all := (others => Float (Random (Gen)));

   D_A := Cuda.Runtime_Api.Malloc (Array_Size);
   D_B := Cuda.Runtime_Api.Malloc (Array_Size);
   D_C := Cuda.Runtime_Api.Malloc (Array_Size);

   Cuda.Runtime_Api.Memcpy
     (Dst   => D_A,
      Src   => H_A.all'Address,
      Count => Array_Size,
      Kind  => Memcpy_Host_To_Device);

   Cuda.Runtime_Api.Memcpy
     (Dst   => D_B,
      Src   => H_B.all'Address,
      Count => Array_Size,
      Kind  => Memcpy_Host_To_Device);

   Put_Line ("CUDA kernel launch with " & blocks_Per_Grid'Img &
               " blocks of " & Threads_Per_Block'Img & "  threads");

   pragma CUDA_Execute (Vector_Add (D_A, D_B, D_C, Num_Elements), Threads_Per_Block, Blocks_Per_Grid);

   Err := Get_Last_Error;

   -- TODO: Need to wrapp correctly Get_Error_String and return a String instead of a C string
   --      if Err /= Success then
   --        Put_Line  ("Failed to launch vectorAdd kernel (error code " & Get_Error_String (Err) & " )");
   --        return;
   --      end if;

   Put_Line ("Copy output data from the CUDA device to the host memory");

   Cuda.Runtime_Api.Memcpy
     (Dst   => H_C.all'Address,
      Src   => D_C,
      Count => Array_Size,
      Kind  => Memcpy_Device_To_Host);

   for I in 1..Num_Elements loop
      if abs (H_A (I) + H_B (I) - H_C (I)) > 1.0E-5 then
         Put_Line ("Result verification failed at element "& I'Img & "!");

         return;
      end if;
   end loop;

   Put_Line ("Test PASSED");

   Free (D_A);
   Free (D_B);
   Free (D_C);

   Free (H_A);
   Free (H_B);
   Free (H_C);

   Put_Line ("Done");
end Main;
