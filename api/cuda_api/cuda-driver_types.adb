with udriver_types_h;
use udriver_types_h;

package body CUDA.Driver_Types is
   ---
   -- Host_Fn_T_Gen --
   ---

   procedure Host_Fn_T_Gen (Arg1 : System.Address) is
      Temp_local_1 : aliased System.Address with
         Address => Arg1'Address,
         Import;

   begin

      declare

      begin
         Temp_Call_1 (Temp_local_1);
         declare

         begin
            null;

            null;
         end;
      end;
   end Host_Fn_T_Gen;

begin
   null;

end CUDA.Driver_Types;
