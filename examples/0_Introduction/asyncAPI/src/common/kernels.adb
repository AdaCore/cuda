with CUDA.Runtime_Api; use CUDA.Runtime_Api;
with Interfaces.C;     use Interfaces.C;

package body Kernels is

   ----------------------
   -- Increment_Kernel --
   ----------------------

   procedure Increment_Kernel
     (G_Data : Array_Device_Access; Inc_Value : Integer)
   is
      Offset : constant Integer :=
        Integer (Block_IDx.X * Block_Dim.X + Thread_IDx.X);
      Idx    : constant Integer := G_Data'First + Offset;
   begin
      G_Data (Idx) := G_Data (Idx) + Inc_Value;
   end Increment_Kernel;

end Kernels;
