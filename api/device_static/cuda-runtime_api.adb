package body CUDA.Runtime_Api is
   function Grid_Dim return CUDA.Vector_Types.Dim3 is
      function Nctaid_X return Interfaces.C.unsigned with
         Inline,
         Import,
         Convention    => C,
         External_Name => "*llvm.nvvm.read.ptx.sreg.nctaid.x";
      function Nctaid_Y return Interfaces.C.unsigned with
         Inline,
         Import,
         Convention    => C,
         External_Name => "*llvm.nvvm.read.ptx.sreg.nctaid.y";
      function Nctaid_Z return Interfaces.C.unsigned with
         Inline,
         Import,
         Convention    => C,
         External_Name => "*llvm.nvvm.read.ptx.sreg.nctaid.z";
   begin
      return (Nctaid_X, Nctaid_Y, Nctaid_Z);
   end Grid_Dim;

   function Block_Idx return CUDA.Vector_Types.Uint3 is
      function Ctaid_X return Interfaces.C.unsigned with
         Inline,
         Import,
         Convention    => C,
         External_Name => "*llvm.nvvm.read.ptx.sreg.ctaid.x";
      function Ctaid_Y return Interfaces.C.unsigned with
         Inline,
         Import,
         Convention    => C,
         External_Name => "*llvm.nvvm.read.ptx.sreg.ctaid.y";
      function Ctaid_Z return Interfaces.C.unsigned with
         Inline,
         Import,
         Convention    => C,
         External_Name => "*llvm.nvvm.read.ptx.sreg.ctaid.z";
   begin
      return (Ctaid_X, Ctaid_Y, Ctaid_Z);
   end Block_Idx;

   function Block_Dim return CUDA.Vector_Types.Dim3 is
      function Ntid_X return Interfaces.C.unsigned with
         Inline,
         Import,
         Convention    => C,
         External_Name => "*llvm.nvvm.read.ptx.sreg.ntid.x";
      function Ntid_Y return Interfaces.C.unsigned with
         Inline,
         Import,
         Convention    => C,
         External_Name => "*llvm.nvvm.read.ptx.sreg.ntid.y";
      function Ntid_Z return Interfaces.C.unsigned with
         Inline,
         Import,
         Convention    => C,
         External_Name => "*llvm.nvvm.read.ptx.sreg.ntid.z";
   begin
      return (Ntid_X, Ntid_Y, Ntid_Z);
   end Block_Dim;

   function Thread_Idx return CUDA.Vector_Types.Uint3 is
      function Tid_X return Interfaces.C.unsigned with
         Inline,
         Import,
         Convention    => C,
         External_Name => "*llvm.nvvm.read.ptx.sreg.tid.x";
      function Tid_Y return Interfaces.C.unsigned with
         Inline,
         Import,
         Convention    => C,
         External_Name => "*llvm.nvvm.read.ptx.sreg.tid.y";
      function Tid_Z return Interfaces.C.unsigned with
         Inline,
         Import,
         Convention    => C,
         External_Name => "*llvm.nvvm.read.ptx.sreg.tid.z";
   begin
      return (Tid_X, Tid_Y, Tid_Z);
   end Thread_Idx;

   function Wrap_Size return Interfaces.C.int is
      function Wrapsize return Interfaces.C.int with
         Inline,
         Import,
         Convention    => C,
         External_Name => "*llvm.nvvm.read.ptx.sreg.wrapsize";
   begin
      return Wrapsize;
   end Wrap_Size;

end CUDA.Runtime_Api;
