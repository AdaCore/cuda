with Kernel_Device; use Kernel_Device;

package body Kernel_Global is
   
   procedure Vector_Add_Global
     (A_Addr : System.Address;
      B_Addr : System.Address;
      C_Addr : System.Address;
      Num_Elements : Integer)
   is
   begin
      Vector_Add_Device (A_Addr, B_Addr, C_Addr, Num_Elements);
   end Vector_Add_Global;

end Kernel_Global;
