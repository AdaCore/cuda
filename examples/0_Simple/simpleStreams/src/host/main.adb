with System.Address_Image;
with Ada.Text_IO; use Ada.Text_IO;
with CUDA.Driver_Types;
with CUDA.Runtime_Api; use CUDA.Runtime_Api;
with CUDA.Vector_Types; use CUDA.Vector_Types;
with udriver_types_h; use udriver_types_h;
with Interfaces; use Interfaces;
with Interfaces.C; use Interfaces.C;
with Interfaces.C.Strings; use Interfaces.C.Strings;

with Support; use Support;
with Kernels; use Kernels;

with Ada.Unchecked_Deallocation;

procedure Main is
   S_SDK_Sample : String := "simpleStreams";

   Cuda_Device : Device_T := 0;
   N_Streams : Integer := 4;
   N_Reps : Integer := 10;
   N : Integer := 16 * 1024 * 1024;
   Threads, Blocks : Dim3;
   Time_Kernel : Float;
   B_Pin_Generic_Memory : Boolean := True; -- DEFAULT_PINNED_GENERIC_MEMORY
   Device_Sync_Method : unsigned := udriver_types_h.cudaDeviceBlockingSync; -- TODO: Needs to be wrapped
   N_Iterations : Integer;
   Device_Properties : CUDA.Driver_Types.Device_Prop;
   C : Integer := 5;

   Device_A : Integer_Array_Device_Access;
   Device_C : Integer_Array_Device_Access;
   Host_A : Integer_Array_Host_Access;

   Streams : array (0 .. N_Streams - 1) of CUDA.Driver_Types.Stream_T;

   Start_Event, Stop_Event : CUDA.Driver_Types.Event_T;
   Event_Flags : unsigned;

   procedure Free is new Ada.Unchecked_Deallocation
     (Integer_Array, Integer_Array_Host_Access);

   procedure Free is new Ada.Unchecked_Deallocation
     (Integer_Array, Integer_Array_Device_Access);

begin
   Put_Line ("[ " & S_SDK_Sample & " ]");
   New_Line;

   Cuda_Device := Get_Device;

   Set_Device (Cuda_Device);
   Device_Properties := Get_Device_Properties (Cuda_Device);
   N_Iterations := 5;

   if B_Pin_Generic_Memory then
      Put_Line
        ("Device" & Cuda_Device'Img & ": canMapHostMemory "
         & To_Ada (Device_Properties.Name)
         & ", "
         & (if Device_Properties.Can_Map_Host_Memory = 0 then "No" else "Yes"));

      if Device_Properties.Can_Map_Host_Memory = 0 then
         Put_Line ("Using cudaMallocHost, CUDA device does not support mapping of generic host memory");

         B_Pin_Generic_Memory := False;
      end if;
   end if;

   Set_Device_Flags (Device_Sync_Method or (if B_Pin_Generic_Memory then cudaDeviceMapHost else 0));

   Host_A := new Integer_Array'(0 .. N - 1 => 0);
   Device_A := new Integer_Array'(0 .. N - 1 => 0);

   -- TODO: This may be slow, we may need to implement a bulk copy instead
   Device_C := new Integer_Array'(0 .. N - 1 => C);

   Put_Line ("Starting Test");

   for I in Streams'Range loop
      Streams (I) := Stream_Create;
   end loop;

   Event_Flags := (if Device_Sync_Method = cudaDeviceBlockingSync then cudaEventBlockingSync else cudaEventDefault);
   Start_Event := Event_Create_With_Flags (Event_Flags);
   Stop_Event := Event_Create_With_Flags (Event_Flags);

   Threads := (512, 1, 1);
   Blocks := (unsigned (N) / Threads.X, 1, 1);
   Event_Record (Start_Event, null);

   pragma CUDA_Execute
     (Init_Array
        (Device_A,
         0, N - 1,
         Device_C,
         N_Iterations),
      Blocks,
      Threads,
      0,
      Streams (0));

   Event_Record (Stop_Event, null);
   Event_Synchronize (Stop_Event);
   Time_Kernel := Event_Elapsed_Time (Start_Event, Stop_Event);

   Put_Line ("kernel:" & Time_Kernel'Img);

   Event_Record (Start_Event, null);

   for K in 1 .. N_Reps loop
      pragma CUDA_Execute
        (Init_Array
           (Device_A,
            0, N - 1,
            Device_C,
            N_Iterations),
         Blocks,
         Threads);
   end loop;

   Event_Record (Stop_Event, null);
   Event_Synchronize (Stop_Event);
   Time_Kernel := Event_Elapsed_Time (Start_Event, Stop_Event);

   Put_Line ("non streamed kernel:" & Float'Image(Time_Kernel / Float (N_Reps)));

   Threads := (512, 1, 1);
   Blocks := (unsigned (N) / (unsigned (Streams'Length) * Threads.X), 1, 1);

   -- TODO: These may be slow, we may need to implement a bulk copy instead
   Host_A.all := (others => 255);
   Device_A.all := (others => 255);

   Event_Record (Start_Event, null);

   declare
      function Low_Bound (Id : Integer) return Integer
      is (Id * N / Streams'Length);

      function High_Bound (Id : Integer) return Integer
      is (Low_Bound (Id + 1) - 1);
   begin
      for K in 1 .. N_Reps loop
         for I in Streams'Range loop
            pragma CUDA_Execute
              (Init_Array
                 (Device_A,
                  Low_Bound (I),
                  High_Bound (I),
                  Device_C,
                  N_Iterations),
               Blocks,
               Threads,
               0,
               Streams (I));
         end loop;

         for I in Streams'Range loop
            Stream_Model.Stream := Streams (I);

            Host_A (I * N / Streams'Length .. (I + 1) * N / Streams'Length - 1) :=
              Device_A (I * N / Streams'Length .. (I + 1) * N / Streams'Length - 1);
         end loop;
      end loop;
   end;

   Event_Record (Stop_Event, null);
   Event_Synchronize (Stop_Event);

   Time_Kernel := Event_Elapsed_Time (Start_Event, Stop_Event);

   Put_Line ("streamed kernel:" & Float'Image(Time_Kernel / Float (N_Reps)));

   declare
      Expected : Integer := C * N_Reps * N_Iterations;
   begin
      for I in 0 .. N - 1 loop
         if Host_A (I) /= Expected then
            Put_Line
              ("ERROR, EXPECTED " & Expected'Img
               & ", GOT" & Host_A (I)'Img
               & " ON" & I'Img);

            exit;
         end if;
      end loop;
   end;

   for I in Streams'Range loop
      Stream_Destroy (Streams (I));
   end loop;

   Event_Destroy (Start_Event);
   Event_Destroy (Stop_Event);

   Free (Host_A);
   Free (Device_A);
   Free (Device_C);
end Main;
