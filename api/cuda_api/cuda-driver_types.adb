with driver_types_h; use driver_types_h;

package body CUDA.Driver_Types is

   procedure Host_Fn_T_Gen (Arg1 : System.Address) is

      Temp_local_1 : aliased System.Address with
         Address => Arg1'Address,
         Import;

   begin
      declare
      begin
         Temp_1 (Temp_local_1);

         declare
         begin
            null;
         end;
      end;
   end Host_Fn_T_Gen;

begin
   null;
end CUDA.Driver_Types;
