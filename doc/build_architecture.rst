**************************************
Build Architecture
**************************************

A GNAT for CUDA application is a distributed application which contains two main
components:

 - A host application, which is responsible for running overall orchestration
   of various kernels to be executed on the device. This application can run
   from either an x86 Linux or an ARM Linux environment.
 - A library of device kernels that can be called from the host application, 
   loaded with the host code.

The host application is compiled as a regular host executable, with special 
switches that identifies that it needs to include specific CUDA instrumentation.

The device library is compiled as a standalone library project. This library
project has run-time restrictions simlar to those that can be found on bare
metal environment with no run-time. For example, Ada tasking is not available.

A typical project contains two projects files, one for each part of the 
application. The first project to be compiled is the device project. A typical
device project file will look like this::

   with "cuda_device.gpr";

   library project Device is

      for Languages use ("Ada");
      for Source_Dirs use ("src/common");
      for Object_Dir use "obj/device";

      for Target use "cuda";
      for Runtime ("ada") use "device-cuda";

      for Create_Missing_Dirs use "True";

      for Library_Name use "kernel";
      for Library_Dir use "lib";
   
      package Compiler is
         for Switches ("ada") use ("-gnatX", "-O2", "-gnatn");      
      end Compiler;
   
   end Device;

A few things are noteworthy here:

 - The target is identified as being ``cuda``. This is what will be needed by
   gprbuild to know which compiler is to be used.
 - The runtime is set to ``device-cuda``. This is the specific restricted
   Ada run-time that contains specifically capabilities available for the 
   device.
 - The compiler switches include ``-O2`` and ``-gnatn``, as to ensure maximum 
   performances on the device.
 - The compiler switches also include ``-gnatX``. While not strictly necessary, 
   a number of capabilites are being added to the Ada programming language
   to make it easier to develop cuda applications.

This project can be easily built with a gprbuild command::

  $> gprbuild -P device.gpr

The result of this is the creation of a library in the form of a fatbinary
under lib/libkernel.fatbin.o. This fatbinary contains the device code ready
to be loaded to the GPU.

An important limitation is that at this stage, a GNAT CUDA application can
only link against one fatbinary at most.

A typical host project file will look like this::

  with "cuda_host.gpr";

  project Host is

      for Exec_Dir use ".";
      for Object_Dir use "obj/host";
      for Source_Dirs use ("src/common", "src/host");
      for Main use ("main.adb");

      for Create_Missing_Dirs use "True";

      package Compiler is
         for Switches ("ada") use ("-gnatX", "-gnatd_c");
      end Compiler;

      package Linker is
         for Switches ("ada") use (
            "-L/usr/lib/cuda/targets/x86_64-linux/lib/stubs", 
            "-L/usr/lib/cuda/targets/x86_64-linux/lib", 
            "-lcudadevrt", 
            "-lcudart_static", 
            "-lrt", 
            "-lpthread", 
            "- ldl",
            "-Wl,--unresolved-symbols=ignore-all");
         for Default_Switches ("ada") use ();
       end Linker;

      package Binder is
         for Default_Switches ("ada") use ("-d_c");
      end Binder;
   end Host;

A few things are noteworthy here:

 - The project depends on cuda_host.gpr, which contains the binding to the CUDA
   API generated during the installation step.
 - The compiler switches contains ``-gnatX`` to enable additional features 
   provided to ease the development of CUDA applications.
 - The compiler switches contains ``-gnatd_c`` to enable CUDA-specific 
   instrumentation.
 - The linker switches includes the list of librairies that needs to be linked
   against to resolve all cuda symbols.
 - The binder switches include ``-d_c`` to enable CUDA-speific instrumentation
   at program initialization

This project can the be build by::

  $> gprbuild -P host.gpr -largs $(PWD)/lib/kernel.fatbin.o 

Note the addition of the fatbinary on the linker line. This comes from the 
previous step.

Once built, the resulting binary can be run similar to any regular binary.