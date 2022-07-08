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

In the extracted directory, generate the toolsuite setup for your current 
installation::

  cd cuda_env-[version]-x86_64-linux-bin
  sh setup.sh

In the same directory, execute::
  
  source ./env.sh

Note that this is needed every time you will need to compile a CUDA application.

This step is only needed once at installation.

You then need to identify the GPU architecture that you're targetting. 
You can get details `here <https://arnon.dk/matching-sm-architectures-arch-and-gencode-for-various-nvidia-cards/>`_.
The following example is assuming sm_75::

  cd cuda/examples/0_Simple/vectorAdd
  make GPU_ARCH=sm_75
  ./main

You should see::

  CUDA kernel launch with  16 blocks of  256  threads
  Copy output data from the CUDA device to the host memory
  Test PASSED
  Done
