**************************************
Installation
**************************************

Before installing, make sure that you have the following prerequisites:

 - An x86_64 Linux environment, with CUDA drivers installed.
 - A GNAT Pro version that is of the same date or more recent than your gnat
   cuda delivery

The following steps are specific for x86 Linux native host. Please refer
to the cross ARM installation if you need to target cross ARM Linux instead.

Then follow these steps

Untar the package::

 tar -xzf tar -xzf cuda_env-[version]-x86_64-linux-bin.tar.gz

In the extracted directory, execute env.sh::

  cd cuda_env-[version]-x86_64-linux-bin
  . env.sh

Note that this is needed every time you will need to compile a CUDA application.

Generate Ada bindings that correspond to your current CUDA installation::

  cd cuda/api
  sh bind.sh
  cd ../..

This step is only needed once at installation.

Try the first example::

  cd cuda/examples/0_Simple/vectorAdd
  make
  ./main

You should see::

  CUDA kernel launch with  16 blocks of  256  threads
  Copy output data from the CUDA device to the host memory
  Test PASSED
  Done
