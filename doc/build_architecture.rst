**************************************
Build Architecture
**************************************

.. role:: switch(samp)

Overall Model
=============

A GNAT for CUDA® application is a distributed application containing two
main components:

- A host application, responsible for the overall orchestration of the
  various kernels to be executed on the device. This application can run
  on either an x86 Linux or an ARM Linux environment.
- A library of device kernels, each containing device code, that can be
  called from the host application.

You compile the host application as a regular host executable, with special
switches that specify that it needs to include specific CUDA operations.

You compile the device library as a standalone library project. This
library project has run-time restrictions similar to those of "bare metal"
environments that contains minimal run-time support. For example, Ada
tasking is not available.

A typical project contains two projects files, one for each component of
the application: the device and the host. A typical device project file
looks like this:

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

A few things to note:

- This project is a standalone Ada project. This means in particular that
  it handles its own elaboration. It's also encapsulated, which means it
  must include all necessary dependencies.
- The project depends on :code:`CUDA_API_Device`, which contains various
  configuration options for CUDA.
- The name of the library is :code:`Device`. You must use that name in the
  default configuration of CUDA projects in the current implementation. You
  could use a different name, but that would require a changes to
  :code:`CUDA_API_Device`.
- You must put all kernels exported from CUDA into the set of units you
  specify in the :code:`Library_Interface` attribute.
- The target is identified as being ``cuda``. This is what :file:`gprbuild`
  needs in order to know which compiler to use.
- The compiler, binder and library switches are coming from the package
  :code:`CUDA_API_Device` and include the specialized switches necessary
  for CUDA.  You can add to these switches, but the binder needs
  :switch:`-d_d=driver` to enable CUDA-specific capabilities and generate a
  driver library that can be elaborated by the host.

You can easily build this project with the :file:`gprbuild` command:

.. code-block:: shell

  gprbuild -P device.gpr

This results in the creation of a library, in the form of a "fat binary",
in :file:`lib/libdevice.fatbin.o`. This fat binary contains the device code
in the format needed for it to be loaded to the GPU.

An important limitation of the current implementation is that a GNAT CUDA
application can only link against at most one fat binary. However, this fat
binary (which is a standalone library) can itself depend on arbitrary number
of static libraries. Note that, in particular, it depends on
:code:`cuda_runtime_api` as well as the run-time.

A typical host project file looks like this:

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

Some things to note here:

- The project depends on :code:`cuda_api_host`, which contains the binding
  to the CUDA API that was generated during the installation step as well
  as various CUDA configuration options.
- The compiler, binder, and linker switches are coming from the package
  :code:`CUDA_API_Device` and include specialized switches necessary for
  CUDA. You can add to these switches, but the compiler needs
  :switch:`-gnatd_c` and the binder needs :code:`-d_c` to enable
  CUDA-specific capabilities.

A current issue in GPRbuild requires ``ADA_INCLUDE_PATH`` to include the CUDA
API path prior to calling the host builder. Note that this same variable
should not be set in the previous step otherwise the driver binding will fail.
Setting up of ``ADA_INCLUDE_PATH`` can be done in the following way, assuming that
``PREFIX`` points to the root directory of your GNAT for CUDA installation:

.. code-block:: shell

  export ADA_INCLUDE_PATH="$PREFIX/api/host/cuda_raw_binding:$PREFIX/api/host/cuda_api:$PREFIX/api/cuda_internal"

This constraint is to be lifted in a future version of the technology.

You can build this project by:

.. code-block:: shell

  gprbuild -P host.gpr -largs $PWD/lib/device.fatbin.o

Note the specification of the fat binary on the linker line. This file was
produced by the previous step.

Once you've built it, the resulting binary can be run in the same way
as any other binary.

You can reuse the standard :file:`makefile` preconfigured in the above way
by including :file:`Makefile.build`, which is located at the top of your
GNAT for CUDA installation, e.g:

.. code-block:: makefile

   include $PREFIX/Makefile.build

   build: gnatcuda_build

Invoking :file:`make` will build the current project. You can look at the
examples shipped with the technology for more details of the actual usage.

Building for Tegra®
===================

Tegra® is an NVIDIA®  SoC that combines ARM cores and NVIDIA GPUs. GNAT
for CUDA® allows you to target this SoC through a cross compiler. The
toolchain is hosted on a x86 64 bits Linux system (the host) and generates
both ARM 64 bits code targeting the Linux environment installed on Tegra®
(the CUDA host) together with the necessary PTX code running over the GPU
(the Device).

To cross-build both CUDA host and device object code from your host
you need:

- This product, GNAT for CUDA
- A GNAT ``aarch64-linux`` cross-compiler toolchain on your host that
  targets the CUDA host.
- The CUDA libraries for the CUDA host. We recommend you access those
  on your host via a network connection to your CUDA host.
- Set the :code:`cuda_host` and :code:`gpu_arch` scenario variables to
  values matching the TEGRA configuration for both the :code:`device`
  and :code:`host` build project. You can find the definition of
  possible values for both scenario variables in
  :file:`architecture.gpr`.
- Finally deploy the built executable to the CUDA host and execute it.

For a detailed set of instructions, please consult the git repository
:file:`README.md` section about `cross-compilation
<https://github.com/AdaCore/cuda#cross-compilation>`_.

Building Examples
=================

You can find examples under the :file:`cuda/examples/` directory. They
are all structured similarly and have:

- two projects at the top level: :file:`device.gpr` for the compilation of
  the device code and :file:`host.gpr` for the compilation of the host code
- a :file:`Makefile` that compiles the whole program and generates an
  executable at the top level
- an :file:`obj/` directory that stores the output of the compilation
  process (automatically generated during the first :file:`make`)
- a :file:`src/` directory that contains the sources of the example

In an example directory, you can make a project with:

.. code-block:: shell

    make

By default, examples are built for the native environment. If you want
to target a cross ARM Linux, you can change the ``CUDA_HOST`` value,
e.g.:

.. code-block:: shell

    make CUDA_HOST=aarch64-linux
