with System;

with Ada.Finalization;

generic
   type T is private;   
package CUDA_Wrapper is

   type T_Access is access all T;
    
   type Wrapper is tagged private;
   function From (From_Val : T) return Wrapper;
   procedure To (Self : Wrapper; Dest : in out T);
   function Get (Self : Wrapper) return T;

   function Device (Self : Wrapper) return T_Access;
   
   type Array_T is array (Natural range <>) of aliased T;
   type Array_Access is access all Array_T;
   type Array_Wrapper is tagged private; 

   procedure Reserve (Self : in out Array_Wrapper; Nb_Elements : Positive);
   function From (From_Val : Array_T) return Array_Wrapper;
   procedure To (Self : Array_Wrapper; Dest : in out Array_T);
   function Device (Self : Array_Wrapper) return Array_Access;
      
private
   
   subtype CUDA_Address is System.Address;
   
   type T_Bounds is record
      First, Last : Natural;
   end record;
   
   type Bounds_Ptr is access all T_Bounds;
      
   type Fat_Pointer is record
      Data   : CUDA_Address;
      Bounds : CUDA_Address;
   end record;  
   
   type Wrapper is new Ada.Finalization.Controlled with record
      Device_Ptr : CUDA_Address;     
   end record;
   
   overriding procedure Finalize (Self : in out Wrapper);
   
   type Array_Wrapper is new Ada.Finalization.Controlled with record
      Device_Ptr : Fat_Pointer;
   end record;   
   
   overriding procedure Finalize (Self : in out Array_Wrapper);
   
end CUDA_Wrapper;
