with Ada.Command_Line;
with Ada.Exceptions;
with Ada.Float_Text_IO;
with Ada.Real_Time; use Ada.Real_Time;
with Ada.Text_IO;
with System;

with Interfaces.C; use Interfaces.C;

with CUDA.Driver_Types;
with CUDA.Exceptions;
with CUDA.Runtime_Api; use CUDA.Runtime_Api;
with CUDA.Stddef;
with CUDA.Storage_Models;

with ucuda_runtime_api_h;
with udriver_types_h;

with Kernels;

procedure Main is
   type Array_Host_Access is access Kernels.Int_Array with
     Designated_Storage_Model => CUDA.Storage_Models.Pagelocked_Model;

   function Correct_Output
     (Data : Array_Host_Access; X : Integer) return Boolean
   is
   begin
      for I in Data'Range loop
         if (Data (I) /= X) then
            Ada.Text_IO.Put_Line
              ("Error! Data (" & I'Image & ") = " & Data (I)'Image &
               ", ref = " & X'Image);
            return False;
         end if;
      end loop;

      return True;
   end Correct_Output;

   -- This function is not present in the original asyncAPI sample. It is
   -- added to encapsulate what we need to do with the thin bindings.
   function Is_Event_Ready (Event : CUDA.Driver_Types.Event_T) return Boolean
   is
   begin
      Event_Query (Event);
      return True;
   exception
      when Cuda.Exceptions.Error_cudaErrorNotReady =>
         return False;
   end Is_Event_Ready;

   CUDA_Device : Device_T := Get_Device;

   N     : constant Integer := 16 * 1_024 * 1_024;
   Value : constant Integer := 26;

   A : Array_Host_Access := new Kernels.Int_Array'(0 .. N - 1 => 0);

   D_A : Kernels.Array_Device_Access := new Kernels.Int_Array (0 .. N - 1);

   Threads_Per_Block : constant Integer := 512;
   Blocks_Per_Grid   : constant Integer := (N - 1) / Threads_Per_Block + 1;

   Start_Event : CUDA.Driver_Types.Event_T := Event_Create;
   Stop_Event  : CUDA.Driver_Types.Event_T := Event_Create;

   Start_Time   : Ada.Real_Time.Time;
   Stop_Time    : Ada.Real_Time.Time;
   Elapsed_Time : Time_Span;

   Counter : Natural := 0;

   GPU_Time : Float;

   package Duration_Text_IO is new Ada.Text_IO.Fixed_IO (Duration);
begin
   Ada.Text_IO.Put_Line
     ("[" & Ada.Command_Line.Command_Name &
      "], with GNAT for CUDA - Starting...");

   -- We set the Stream component to `null` to give it the "default stream"
   -- meaning of the CUDA API.
   Kernels.Stream_Model.Stream := null;

   declare
      Device_Properties : CUDA.Driver_Types.Device_Prop :=
        Get_Device_Properties (CUDA_Device);
   begin
      Ada.Text_IO.Put_Line
        ("CUDA device [" & Interfaces.C.To_Ada (Device_Properties.Name) & "]");
   end;

   declare
      Length : CUDA.Stddef.Size_T := D_A.all'Size / Interfaces.C.CHAR_BIT;
   begin
      MemSet (System.Address (D_A.all'Address), 255, Length);
   end;

   Device_Synchronize;

   Start_Time := Ada.Real_Time.Clock;

   -- In the call below, we use `null` to use the default stream, like `0` is
   -- used in the original example. This might warrant adding a constant later.
   Event_Record (Start_Event, null);

   D_A.all := A.all;

   pragma Cuda_Execute
     (Kernels.Increment_Kernel (D_A, Value),
      Blocks_Per_Grid,
      Threads_Per_Block,
      0,
      null);

   A.all := D_A.all;

   Event_Record (Stop_Event, null);

   Stop_Time := Ada.Real_Time.Clock;

   while not Is_Event_Ready (Stop_Event) loop
      Counter := Counter + 1;
   end loop;

   GPU_Time := Event_Elapsed_Time (Start_Event, Stop_Event);

   Elapsed_Time := Stop_Time - Start_Time;

   Ada.Text_IO.Put ("time spent executing by the GPU: ");
   Ada.Float_Text_IO.Put (GPU_Time, Fore => 1, Aft => 2, Exp => 0);
   Ada.Text_IO.New_Line;

   Ada.Text_IO.Put ("time spent by CPU in CUDA calls: ");
   Duration_Text_IO.Put
     (To_Duration (Elapsed_Time), Fore => 1, Aft => 2, Exp => 0);
   Ada.Text_IO.New_Line;

   Ada.Text_IO.Put_Line
     ("CPU executed " & Counter'Image &
      " Iterations while waiting for GPU to finish");

   if Correct_Output (A, Value) then
      Ada.Command_Line.Set_Exit_Status (Ada.Command_Line.Success);
   else
      Ada.Command_Line.Set_Exit_Status (Ada.Command_Line.Failure);
   end if;
end Main;
