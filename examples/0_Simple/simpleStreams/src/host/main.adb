with System.Address_Image;
with Ada.Text_IO; use Ada.Text_IO;
with CUDA.Driver_Types;
with CUDA.Runtime_Api; use CUDA.Runtime_Api;
with CUDA.Vector_Types; use CUDA.Vector_Types;
with udriver_types_h; use udriver_types_h;
with Interfaces; use Interfaces;
with Interfaces.C; use Interfaces.C;
with Interfaces.C.Strings; use Interfaces.C.Strings;

with Storage_Models;
with Storage_Models.Arrays;

with CUDA_Storage_Models; use CUDA_Storage_Models;

with Support; use Support;
with Kernels; use Kernels;

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

   package Device_Int_Array is new CUDA_Storage_Models.Malloc_Storage_Model.Arrays
     (Integer, Natural, Integer_Array, Integer_Array_Access);

   package Host_Int_Array is new CUDA_Storage_Models.Malloc_Host_Storage_Model.Arrays
     (Integer, Natural, Integer_Array, Integer_Array_Access);

   use Device_Int_Array;
   use Host_Int_Array;

   Device_A : Device_Int_Array.Foreign_Array_Access;
   Device_C : Device_Int_Array.Foreign_Array_Access;
   Host_A : Host_Int_Array.Foreign_Array_Access;

   Streams : array (0 .. N_Streams - 1) of CUDA.Driver_Types.Stream_T;
   Device_A_Slices : array (0 .. N_Streams - 1) of Device_Int_Array.Foreign_Array_Slice_Access;
   Host_A_Slices : array (0 .. N_Streams - 1) of Host_Int_Array.Foreign_Array_Slice_Access;

   Start_Event, Stop_Event : CUDA.Driver_Types.Event_T;
   Event_Flags : unsigned;

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

   Host_A := Allocate (0, N - 1);
   Assign (Host_A, 0);

   Device_A := Allocate (0, N - 1);
   Assign (Device_A, 0);
   Device_C := Allocate (0, N - 1);
   Assign (Device_C, C);

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
        (Uncheck_Convert (Device_A),
         Uncheck_Convert (Device_C),
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
           (Uncheck_Convert (Device_A),
            Uncheck_Convert (Device_C),
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
   Assign (Host_A, 255);
   Assign (Device_A, 0);
   Event_Record (Start_Event, null);

   for I in Streams'Range loop
      Device_A_Slices (I) := Allocate (Device_A, I * N / Streams'Length, N - 1);
      Host_A_Slices (I) := Allocate (Host_A, I * N / Streams'Length, N - 1);
   end loop;

   for K in 1 .. N_Reps loop
      for I in Streams'Range loop
         pragma CUDA_Execute
           (Init_Array
              (Uncheck_Convert (Device_A_Slices (I)),
               Uncheck_Convert (Device_C),
               N_Iterations),
            Blocks,
            Threads,
            0,
            Streams (I));
      end loop;

      for I in Streams'Range loop
         Assign (Uncheck_Convert (Host_A_Slices (I)).all,
                 Device_A_Slices (I),
                 (True, Streams (I)));
      end loop;
   end loop;

   Event_Record (Stop_Event, null);
   Event_Synchronize (Stop_Event);

   Time_Kernel := Event_Elapsed_Time (Start_Event, Stop_Event);

   Put_Line ("streamed kernel:" & Float'Image(Time_Kernel / Float (N_Reps)));

   declare
      Expected : Integer := C * N_Reps * N_Iterations;
   begin
      for I in 0 .. N - 1 loop
         if Uncheck_Convert (Host_A) (I) /= Expected then
            Put_Line
              ("ERROR, EXPECTED " & Expected'Img
               & ", GOT" & Uncheck_Convert (Host_A)(I)'Img
               & " ON" & I'Img);

            exit;
         end if;
      end loop;
   end;

   for I in Streams'Range loop
      Stream_Destroy (Streams (I));
      Deallocate (Device_A_Slices (I));
      Deallocate (Host_A_Slices (I));
   end loop;

   Event_Destroy (Start_Event);
   Event_Destroy (Stop_Event);

   Deallocate (Host_A);
   Deallocate (Device_A);
   Deallocate (Device_C);
end Main;
