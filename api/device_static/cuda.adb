with Interfaces.C.Strings;

package body CUDA is

  procedure Last_Chance_Handler
     (File : System.Address; Line : Integer)
  is
   procedure Assert_Fail
     (Assertion : Interfaces.C.char_array;
      File : System.Address;
      Line : Interfaces.C.unsigned;
      Func : Interfaces.C.char_array;
      CharSize : Interfaces.C.size_t)
    with Import,
      Convention => C,
      Link_Name => "__assertfail";
  begin
     Assert_Fail (
        (1 => interfaces.C.char (ASCII.nul)),
         File,
         Interfaces.C.unsigned (Line),
         (1 => interfaces.C.char (ASCII.nul)),
         1);
  end Last_Chance_Handler;

end CUDA;
