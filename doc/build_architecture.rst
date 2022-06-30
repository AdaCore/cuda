**************************************
Build Architecture
**************************************

Overall Model
=============

A GNAT for CUDA® application is a distributed application which contains two main
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
device project file will look like this:

.. code-block:: ada

   with "cuda_api_device.gpr";

   library project Device is

      for Languages use ("Ada");
      for Source_Dirs use ("src/common");
      for Object_Dir use "obj/device";

      for Target use "cuda";
      for Library_Name use "kernel";
      for Library_Dir use "lib";

      package Compiler is
         for Switches ("ada") use CUDA_API_Device.Compiler'Switches ("ada");      
      end Compiler;

      for Archive_Builder use CUDA_API_Device'Archive_Builder;
   
   end Device;

A few things are noteworthy here:

 - The project depends on ``cuda_api_device.gpr``, which contains various configuration
   elements related to CUDA.
 - The target is identified as being ``cuda``. This is what will be needed by
   gprbuild to know which compiler is to be used.
 - The compiler switches are coming from the package CUDA_API_Device, and 
   include specialized switches necessary for CUDA. User can add to these 
   switches.
 - The archive builder is coming from CUDA_API_Device.

This project can be easily built with a gprbuild command::

  $> gprbuild -P device.gpr -Xgpu_arch=sm_75

Note the additional parameter on the command line ``-Xgpu_arch=sm_75``. It is
necessary to specify the GPU architecture for which you're compiling to. In 
this case, ``sm_75`` corresponds to the Turning family of GPUs. You will want
to adjust depending on the actual hardware that you're targetting. Supported 
options are docmented in the project ``cuda_api_device.gpr``.

The result of this is the creation of a library in the form of a fatbinary
under lib/libkernel.fatbin.o. This fatbinary contains the device code ready
to be loaded to the GPU.

An important limitation is that at this stage, a GNAT CUDA application can
only link against one fatbinary at most.

A typical host project file will look like this:

.. code-block:: ada

   with "cuda_api_host.gpr";

   project Host is

      for Exec_Dir use ".";
      for Object_Dir use "obj/host";
      for Source_Dirs use ("src/**");
      for Main use ("main.adb");
   
      for Target use CUDA_API_Host.CUDA_Host;

      package Compiler is
         for Switches ("ada") use  CUDA_API_Host.Compiler'Switches ("ada");
      end Compiler;

      package Linker is
         for Switches ("ada") use CUDA_API_Host.Linker'Switches ("ada");
      end Linker;

      package Binder is
          for Default_Switches ("ada") use CUDA_API_Host.Binder'Default_Switches ("ada");
      end Binder;
   end Host;

A few things are noteworthy here:

 - The project depends on ``cuda_api_host.gpr``, which contains the binding to the CUDA
   API generated during the installation step as well as various configuration
   elements related to CUDA.
 - The compiler, binder and linker switches are coming from the package 
   CUDA_API_Device, and include specialized switches necessary for CUDA. User
   can add to these switches. Note that amongst these switches, the compiler
   needs ``-gnatd_c`` and the binder ``-d_c`` in order to enable CUDA specific 
   capbilities.

This project can the be build by::

  $> gprbuild -P host.gpr -largs $(PWD)/lib/kernel.fatbin.o 

Note the addition of the fatbinary on the linker line. This comes from the 
previous step.

Once built, the resulting binary can be run similar to any regular binary.

Building for Tegra®
===================

Tegra® is an NVIDIA SoC that conbines together ARM cores and NVIDIA GPUs. GNAT
for CUDA® allow to target this SoC through a cross compiler. The toolchain is
hosted on a x86 64 bits Linux system and will generate both ARM 64 bits code
targeting the Linux environment installed on Tegra together with the necessary
PTX code.

TODO