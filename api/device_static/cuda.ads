with System;

package CUDA is

  procedure Last_Chance_Handler
     (File : System.Address; Line : Integer);
  pragma Export (C, Last_Chance_Handler, "__gnat_last_chance_handler");

end CUDA;
