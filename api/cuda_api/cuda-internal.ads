with System;
with Interfaces.C.Strings;
with CUDA.Vector_Types;
with CUDA.Crtdefs;
with CUDA.Driver_Types;

package CUDA.Internal is

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
      (Grid_Dim : Vector_Types.Dim3;
       Block_Dim : Vector_Types.Dim3;
       Shared_Mem : Crtdefs.Size_T;
       Stream : Driver_Types.Stream_T)
       with Import => True,
            Convention => C,
            External_Name => "__cudaPushCallConfiguration";

   procedure Pop_Call_Configuration 
      --  (Grid_Dim : vector_types.dim3;
      --   Block_Dim : vector_types.dim3;
      --   Shared_Mem : crtdefs.Size_T;
      --   Stream : Driver_Types.Stream_T)
      (Grid_Dim : System.Address;
       Block_Dim : System.Address;
       Shared_Mem : System.Address;
       Stream : System.Address)
       with Import => True,
            Convention => C,
            External_Name => "__cudaPopCallConfiguration";

end CUDA.Internal;
