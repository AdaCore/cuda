generic
   type T is private;   
package CUDA_Wrapper is

   type T_Access is access all T;
       
   type Array_T is array (Natural range <>) of aliased T;
   type Array_Access is access all Array_T;     
   
end CUDA_Wrapper;
