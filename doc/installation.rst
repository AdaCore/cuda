**************************************
Prerequisites and environment setup
**************************************

Before installing this software, make sure you have the pre-requisites
corresponding to your build environment installed.

Before going to platform-specific prerequisites, download CUDA Toolkit for
your development host from https://developer.nvidia.com/cuda-downloads.
This is required for building the CUDA bindings.

You need to have the CUDA Toolkit in your PATH, and in particular ptxas.
You can check that by running:

.. code-block:: shell

  which ptxas

If it doesn't return anything, CUDA may not be installed, or need to be
put in your PATH, e.g.:

.. code-block:: shell

   export PATH=/usr/local/cuda-<your CUDA version>/bin:$PATH

Native compiler on x86_64 Linux
**************************************

In case both the development host and the target are running x86_64 Linux
then the following tools are required:

 - An x86_64 Linux environment with CUDA drivers (see above)
 - An installation of GNAT Pro, version 24.0w (20230413) or later.

Cross compilation for aarch64 Linux
**************************************

If the development host is running x86_64 Linux and the target
aarch64 Linux then the following tools are required:

 - An aarch64 Linux environment with CUDA drivers on the target
 - An installation of GNAT Pro cross toolchain for aarch64-linux, 
   version 24.0w (20230413) or later, on the development host.

Obtain a copy of the system libraries according to the instructions 
in the cross toolchain documentation and place them in a on your choice.

As an example, the files can be copied form the target board as follows:

.. code-block:: shell

  $ mkdir ./sysroot
  $ mkdir ./sysroot/usr
  $ scp -rp <my-aarch64-linux-target>:/usr/include ./sysroot/usr/
  $ scp -rp <my-aarch64-linux-target>:/usr/lib ./sysroot/usr/
  $ scp -rp <my-aarch64-linux-target>:/usr/lib64 ./sysroot/usr/
  $ scp -rp <my-aarch64-linux-target>:/lib ./sysroot/
  $ scp -rp <my-aarch64-linux-target>:/lib64 ./sysroot/

Obtain a copy of the CUDA libraries from the target board and place it 
in the targets folder of your CUDA setup:

.. code-block:: shell

  $ scp -rp jetty:/usr/local/cuda/targets/aarch64-linux ./
  $ sudo mv aarch64-linux /usr/local/cuda/targets

Make the sysroot location visible to GNAT via the `ENV_PREFIX` variable

.. code-block:: shell

  $ export ENV_PREFIX=`pwd`/sysroot

Let the toolchain know, that the intended compilation target is aarch64-linux

.. code-block:: shell

  $ export cuda_host=aarch64-linux

**************************************
GNAT-CUDA setup
**************************************

After setting up the environment, you can extract the gnat-cuda package:

.. code-block:: shell

   tar -xzf gnat-cuda-[version]-x86_64-linux-bin.tar.gz

Now you need to know which GPU architecture you're targeting. This is
typically an ``sm``\_ prefix followed by a number. For example
``sm_89`` is the Ada Lovelace architecture. You can find details `on
the GPU architecture mapping here
<https://arnon.dk/matching-sm-architectures-arch-and-gencode-for-various-nvidia-cards/>`_.
You pass this parameter to the next script.

In the extracted directory, generate the tool suite setup for your
current installation:

.. code-block:: shell

  cd gnat-cuda-[version]-x86_64-linux-bin/cuda
  ./setup.sh -mcpu <your GPU architecture>

In the same directory, execute:

.. code-block:: shell

  source ./env.sh

You need to perform the above step every time you want to compile a
CUDA application.

To check if everything is correctly installed, you can try an example:

.. code-block:: shell

  cd cuda/examples/0_Introduction/vectorAdd
  make
  ./main

.. note:: 
  In cross compilation workflow you have to copy `main` to target
  before executing it

You need only perform this check at installation. You should see:

.. code-block:: shell

  CUDA kernel launch with  16 blocks of  256  threads
  Copy output data from the CUDA device to the host memory
  Test PASSED
  Done
