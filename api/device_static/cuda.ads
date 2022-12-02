with Interfaces.C.Strings;

package CUDA is

  procedure Last_Chance_Handler
     (File : Interfaces.C.Strings.char_array_access; Line : Integer);
  pragma Export (C, Last_Chance_Handler, "__gnat_last_chance_handler");

end CUDA;
