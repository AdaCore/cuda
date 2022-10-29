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

At this stage, you need to know which GPU architecture you're targeting. This
will typically be a sm\_ prefix followed by a number, for example sm_89 is the
Ada Lovelace architecture. You can find details
`on this GPU architecture mapping <https://arnon.dk/matching-sm-architectures-arch-and-gencode-for-various-nvidia-cards/>`_.
This parameter is to be passed to the next script.

In the extracted directory, generate the toolsuite setup for your current
installation::

  cd cuda_env-[version]-x86_64-linux-bin/cuda
  sh setup.sh -mcpu <your GPU architecture>

In the same directory, execute::

  source ./env.sh

Note that this is needed every time you will need to compile a CUDA application.

This step is only needed once at installation.

To check if everything is correctly installed, you can try an example:

  cd cuda/examples/0_Simple/vectorAdd
  make
  ./main

You should see::

  CUDA kernel launch with  16 blocks of  256  threads
  Copy output data from the CUDA device to the host memory
  Test PASSED
  Done
