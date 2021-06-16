with System;

package Kernel_Device is

   type Float_Array is array (Integer range <>) of Float;

   type Access_Host_Float_Array is access all Float_Array;

   procedure Vector_Add_Device
     (A_Addr : System.Address;
      B_Addr : System.Address;
      C_Addr : System.Address;
      Num_Elements : Integer);

end Kernel_Device;
