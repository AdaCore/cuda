with Device_Functions; use Device_Functions;

package body Kernel is
   
   procedure Vector_Add
     (A_Addr : System.Address;
      B_Addr : System.Address;
      C_Addr : System.Address;
      Num_Elements : Integer)
   is
   begin
      Vector_Add_Device (A_Addr, B_Addr, C_Addr, Num_Elements);
   end Vector_Add;

end Kernel;
