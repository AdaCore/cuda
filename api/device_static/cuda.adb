package body CUDA is

  procedure Last_Chance_Handler
     (File : Interfaces.C.Strings.char_array_access; Line : Integer)
  is
   procedure Assert_Fail
     (Assertion : Interfaces.C.char_array;
      File : Interfaces.C.char_array;
      Line : Interfaces.C.unsigned;
      Func : Interfaces.C.char_array;
      CharSize : Interfaces.C.size_t)
    with Import,
      Convention => C,
      Link_Name => "__assertfail";
  begin
     Assert_Fail (
        (1 => interfaces.C.char (ASCII.nul)),
         File.all,
         Interfaces.C.unsigned (Line),
         (1 => interfaces.C.char (ASCII.nul)),
         1);
  end Last_Chance_Handler;

end CUDA;
