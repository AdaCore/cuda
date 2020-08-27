with System;
with Interfaces.C;
with Interfaces.C.Strings;
with CUDA.Stddef;
with CUDA.Driver_Types;

package CUDA.Internal is

   type Dim3 is record
      X : Interfaces.C.Unsigned;
      Y : Interfaces.C.Unsigned;
      Z : Interfaces.C.Unsigned;
   end record with Convention => C_Pass_By_Copy;

   type Fatbin_Wrapper is record
      Magic : Interfaces.C.int;
      Version : Interfaces.C.int;
      Data : System.Address;
      Filename_Or_Fatbins : System.Address;
   end record;

   procedure Launch_Kernel
     (Func       : System.Address;
      Grid_Dim   : Dim3;
      Block_Dim  : Dim3;
      Args       : System.Address;
      Shared_Mem : Interfaces.C.Unsigned_Long;
      Stream     : CUDA.Driver_Types.Stream_T)
   with Import => True,
        Convention => C,
        External_Name => "cudaLaunchKernel";

   procedure Register_Function
      (Fat_Binary_Handle : System.Address;
       Func : System.Address;
       Kernel_Name : Interfaces.C.Strings.chars_ptr;
       Kernel_Name_2 : Interfaces.C.Strings.chars_ptr;
       Minus_One : Integer;
       Nullptr1 : System.Address;
       Nullptr2 : System.Address;
       Nullptr3 : System.Address;
       Nullptr4 : System.Address;
       Nullptr5 : System.Address)
      with Import => True,
           Convention => C,
           External_Name => "__cudaRegisterFunction";

   function Register_Fat_Binary (Fat_Binary : System.Address)
      return System.Address
      with Import => True,
           Convention => C,
           External_Name => "__cudaRegisterFatBinary";

   procedure Register_Fat_Binary_End (Fat_Binary : System.Address)
      with Import => True,
           Convention => C,
           External_Name => "__cudaRegisterFatBinaryEnd";

   procedure Push_Call_Configuration
      (Grid_Dim : Dim3;
       Block_Dim : Dim3;
       Shared_Mem : Stddef.Size_T;
       Stream : Driver_Types.Stream_T)
       with Import => True,
            Convention => C,
            External_Name => "__cudaPushCallConfiguration";

   procedure Pop_Call_Configuration
      (Grid_Dim : System.Address;
       Block_Dim : System.Address;
       Shared_Mem : System.Address;
       Stream : System.Address)
       with Import => True,
            Convention => C,
            External_Name => "__cudaPopCallConfiguration";

end CUDA.Internal;
