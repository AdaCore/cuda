with System;
with Interfaces.C;              use Interfaces.C;

with Ada.Numerics.Float_Random; use Ada.Numerics.Float_Random;
with Ada.Text_IO;               use Ada.Text_IO;

with CUDA.Driver_Types;         use CUDA.Driver_Types;
with CUDA.Runtime_Api;          use CUDA.Runtime_Api;
with CUDA.Stddef;
with CUDA.Storage_Models;        use CUDA.Storage_Models;

with Kernel; use Kernel;

with Ada.Unchecked_Deallocation;
with Ada.Unchecked_Conversion;

procedure Main is

   type Array_Host_Access is access all Float_Array;

   procedure Free is new
     Ada.Unchecked_Deallocation (Float_Array, Array_Host_Access);

   procedure Free is new Ada.Unchecked_Deallocation
     (Float_Array, Array_Device_Access);

   Num_Elements : Integer := 4096;

   H_A, H_B, H_C : Array_Host_Access;
   D_A, D_B, D_C : Array_Device_Access;
   Array_Size : CUDA.Stddef.Size_T := CUDA.Stddef.Size_T(Interfaces.C.c_float'size * Num_Elements / 8);

   Threads_Per_Block : Integer := 256 + 1;
   Blocks_Per_Grid : Integer :=
     (Num_Elements + Threads_Per_Block - 1) / Threads_Per_Block;

   Gen : Generator;
   Err : Error_T;

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

   D_A := new Float_Array'(H_A.all);
   D_B := new Float_Array'(H_B.all);
   D_C := new Float_Array (H_C.all'Range);

   Put_Line ("CUDA kernel launch with " & blocks_Per_Grid'Img &
               " blocks of " & Threads_Per_Block'Img & "  threads");

   pragma CUDA_Execute
     (Vector_Add (D_A, D_B, D_C),
      Threads_Per_Block,
      Blocks_Per_Grid);

   Put_Line ("Copy output data from the CUDA device to the host memory");

   H_C.all := D_C.all;
end Main;
