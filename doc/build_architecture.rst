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

   with "cuda_api_device";

   library project Device is
      for Languages use ("Ada");

      for Target use "cuda";
      for Library_Name use "device";
      for Library_Dir use "lib";
      for Library_Kind use "dynamic";
      for Library_Interface use ("kernel");
      for Library_Standalone use "encapsulated";

      package Compiler is
         for Switches ("ada") use CUDA_API_Device.Compiler_Options;
      end Compiler;

      package Binder is
         for Default_Switches ("ada") use CUDA_API_Device.Binder_Options;
      end Binder;

      for Library_Options use CUDA_API_Device.Library_Options;
   end Device;

A few things are noteworthy here:

- This project is a standalone Ada project. This means in particular that it
  can handle its own elaboration. It's also encapsulated, which means that it
  will include all the necessary dependencies.
- The name of that library is ``device``. This is mandatory in the default
  configuration of CUDA projects at this stage. Using a different name is possible
  but requires different setup on ``cuda_api_device``.
- Kernels exported from CUDA must be in the closure of units provided to the
  attribute ``Library_Interface``.
- The project depends on ``cuda_api_device``, which contains various configuration
  elements related to CUDA.
- The target is identified as being ``cuda``. This is what will be needed by
  gprbuild to know which compiler is to be used.
- The compiler, binder and library switches are coming from the package
  CUDA_API_Device, and include specialized switches necessary for CUDA.
  User can add to these switches. Note that amongst these switches, the
  the binder needs ``-d_d=driver`` in order to enable CUDA specific
  capbilities and generate a driver library that can be elaborated by the
  host.

This project can be easily built with a gprbuild command::

  gprbuild -P device.gpr

The result of this is the creation of a library in the form of a fatbinary
under lib/libdevice.fatbin.o. This fatbinary contains the device code ready
to be loaded to the GPU.

An important limitation is that at this stage, a GNAT CUDA application can
only link against one fatbinary at most. However, this standalone library
can itself depend on arbitrary number of static library (and effectively
in this specific case, it depends on cuda_runtime_api as well as the run-time).

A typical host project file will look like this:

.. code-block:: ada

   with "cuda_api_host.gpr";

   project Host is
      for Main use ("main.adb");

      for Target use CUDA_API_Host.CUDA_Host;

      package Compiler is
         for Switches ("ada") use  CUDA_API_Host.Compiler_Options;
      end Compiler;

      package Linker is
         for Switches ("ada") use CUDA_API_Host.Linker_Options;
      end Linker;

      package Binder is
        for Default_Switches ("ada") use CUDA_API_Host.Binder_Options;
      end Binder;
   end Host;

A few things are noteworthy here:

- The project depends on ``cuda_api_host``, which contains the binding to the CUDA
  API generated during the installation step as well as various configuration
  elements related to CUDA.
- The compiler, binder and linker switches are coming from the package
  CUDA_API_Device, and include specialized switches necessary for CUDA. User
  can add to these switches. Note that amongst these switches, the compiler
  needs ``-gnatd_c`` and the binder ``-d_c`` in order to enable CUDA specific
  capbilities.

This project can the be build by::

  gprbuild -P host.gpr -largs $PWD/lib/device.fatbin.o

Note the addition of the fatbinary on the linker line. This comes from the
previous step.

Once built, the resulting binary can be run similar to any regular binary.

A standard makefile preconfigured for the above model can be reused. You can
include Makefile.build which is located at the root of your gnat for cuda
installation, e.g:

.. code-block:: makefile

   include $GNAT_FOR_CUDA_PREFIX/Makefile.build

   build: gnatcuda_build

Invoking make will build the current project. You can look at the examples
shipped with the technology for actual usage.

Building for Tegra®
===================

Tegra® is an NVIDIA® SoC that combines together ARM cores and NVIDIA GPUs. GNAT
for CUDA® allow to target this SoC through a cross compiler. The toolchain is
hosted on a x86 64 bits Linux system (Host) and will generate both ARM 64 bits code
targeting the Linux environment installed on Tegra® (CUDA_Host) together with the necessary
PTX code running over the GPU (Device).

To cross build valid (CUDA_Host and Device) object code from your (Host) you will need:

- This project, GNAT for CUDA: A specialized Ada runtime leveraging Ada bindings for CUDA running on the (Device).
- A GNAT ``aarch64-linux`` cross-compiler toolchain on your (Host) and targeting the (CUDA_Host).
- The CUDA libraries of the (CUDA_Host). We recommend you network access those installed on your (CUDA_Host) from the (HOST).
- Set the ``cuda_host`` and ``gpu_arch`` scenario variables matching the TEGRA configuration for both the ``device`` and ``host`` build project. The definition of possible values for both scenario variables are found in ``cuda_api_device.gpr``.
- Finally deploy to, then execute the built executable at the (CUDA_HOST).

For a detailed and contextualized instruction set, please consult the git repository front-facing README.md section about `cross-compilation <https://github.com/AdaCore/cuda#cross-compilation>`_.

Building Examples
=================

Examples are located under the ``cuda/examples/`` directory. They are all
structured more or less the same:

- two projects at the root level, ``device.gpr`` that controls the device code
  compilation, and ``host.gpr`` that controls host code compilation.
- a Makefile that compiles the whole program and generates a main at the root
- an obj/ directory to store the output of the compilation process (automatically
  generated during the first make)
- a src/ directory that contains sources

In an example directory, a project can be made by the following command::

    make

By default, examples are built for the native environment. If you want to target
a cross ARM Linux, you can also change the ``CUDA_HOST`` value, e.g.::

    make CUDA_HOST=aarch64-linux