with Interfaces.C; use Interfaces.C;
with Interfaces.C.Strings;
with CUDA.Internal;
with CUDA.GPU_Api;  use CUDA.GPU_Api;

package body Kernel is
   
   procedure Vector_Add
     (A_Addr : System.Address;
      B_Addr : System.Address;
      C_Addr : System.Address;
      Num_Elements : Integer)
   is
      A : Float_Array (1..Num_Elements) with Address => A_Addr;
      B : Float_Array (1..Num_Elements) with Address => B_Addr;
      C : Float_Array (1..Num_Elements) with Address => C_Addr;
      I : Integer := Integer (Block_Dim.X * Block_IDx.X + Thread_IDx.X);
   begin
      if I < Num_Elements then
         C (C'First + I) := A (A'First + I) + B (B'First + I);
      end if;
   end Vector_Add;

   --  procedure Initialize_Cuda_Kernel is
   --     Fat_Binary : System.Address;
   --     pragma Import
   --       (Convention    => C,
   --        Entity        => Fat_Binary,
   --        External_Name => "_binary_kernel_fatbin_start" );

   --     type Fatbin_Wrapper is record
   --        Magic : Interfaces.C.int;
   --        Version : Interfaces.C.int;
   --        Data : System.Address;
   --        Filename_Or_Fatbins : System.Address;
   --     end record;

   --     Vector_Add_Name : Interfaces.C.Strings.Chars_Ptr
   --        := Interfaces.C.Strings.New_Char_Array("kernel__vector_add");

   --     Wrapper : Fatbin_Wrapper
   --        := (16#466243b1#, 1, Fat_Binary'Address, System.Null_Address);

   --     Fat_Binary_Handle : System.Address
   --        := CUDA.Internal.Register_Fat_Binary (Wrapper'Address);
   --  begin
   --     CUDA.Internal.Register_Function
   --        (Fat_Binary_Handle,
   --         Vector_Add'Address,
   --         Vector_Add_Name,
   --         Vector_Add_Name,
   --         -1,
   --         System.Null_Address,
   --         System.Null_Address,
   --         System.Null_Address,
   --         System.Null_Address,
   --         System.Null_Address);
   --     CUDA.Internal.Register_Fat_Binary_End (Fat_Binary_Handle);
   --  end;

   --  Initialized_Cuda_Kernel : Boolean := Initialize_Cuda_Kernel;

end Kernel;
