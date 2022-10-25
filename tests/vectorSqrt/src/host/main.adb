with System;
with Interfaces.C;              use Interfaces.C;

with Ada.Numerics.Float_Random; use Ada.Numerics.Float_Random;
with Ada.Text_IO;               use Ada.Text_IO;

with CUDA.Driver_Types;         use CUDA.Driver_Types;
with CUDA.Runtime_Api;          use CUDA.Runtime_Api;
with CUDA.Stddef;

with Kernel; use Kernel;

with Ada.Unchecked_Deallocation;
with Ada.Numerics; use Ada.Numerics;
with Ada.Numerics.Generic_Elementary_Functions;

procedure Main is
   package Elementary_Functions is new
      Ada.Numerics.Generic_Elementary_Functions (Float);

   Num_Elements : Integer := 512;

   H_A, H_B : Access_Host_Float_Array;
   D_A, D_B : System.Address;
   Array_Size : CUDA.Stddef.Size_T := CUDA.Stddef.Size_T(Interfaces.C.c_float'size * Num_Elements / 8);

   Threads_Per_Block : Integer := 256;
   Blocks_Per_Grid : Integer :=
     (Num_Elements + Threads_Per_Block - 1) / Threads_Per_Block;

   Gen : Generator;
   Err : Error_T;

   procedure Free is new
     Ada.Unchecked_Deallocation (Float_Array, Access_Host_Float_Array);

begin
   Put_Line ("[Vector sqrt of " & Num_Elements'Img & " elements]");

   H_A := new Float_Array (1 .. Num_Elements);
   H_B := new Float_Array (1 .. Num_Elements);

   if H_A = null or else H_B = null then
      Put_Line("Failed to allocate host vectors!");

      return;
   end if;

   for I in H_A'First .. H_A'Last loop
     H_A (I) := FLoat(I * I);
   end loop;

   D_A := Cuda.Runtime_Api.Malloc (Array_Size);
   D_B := Cuda.Runtime_Api.Malloc (Array_Size);

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

   pragma CUDA_Execute (Vector_Sqrt (D_A, D_B, Num_Elements), Threads_Per_Block, Blocks_Per_Grid);

   Err := Get_Last_Error;

   Put_Line ("Copy output data from the CUDA device to the host memory");

   Cuda.Runtime_Api.Memcpy
     (Dst   => H_B.all'Address,
      Src   => D_B,
      Count => Array_Size,
      Kind  => Memcpy_Device_To_Host);


   for I in 1..Num_Elements loop
      if H_B (I) - Elementary_Functions.Sqrt(H_A(I)) > 1.0 then
         Put_Line ("Result verification failed at element "& I'Img & "!");

         return;
      end if;
   end loop;

   Put_Line ("Test PASSED");

   Free (D_A);
   Free (D_B);

   Free (H_A);
   Free (H_B);

   Put_Line ("Done");
end Main;
