**************************************
Installation
**************************************

Before installing this software, make sure you have the following
prerequisites installed:

 - An x86_64 Linux environment with CUDA drivers
 - An installation of GNAT Pro, version 24.0w (20230413) or later.

Use the following steps for an x86 Linux native host. Please refer to
the cross ARM installation if you need to target cross ARM Linux
instead.

Prior to running GNAT for CUDA, you need to have the NVIDIA environment
in your PATH, and in particular ptxas. You can check that by running:

.. code-block:: shell

  which ptxas

If it doesn't return anything, CUDA may not be installed, or need to be
put in your PATH, e.g.:

.. code-block:: shell

   export PATH=/usr/local/cuda-<your CUDA version>/bin/:$PATH

From there, you can untar the package:

.. code-block:: shell

   tar -xzf cuda_env-[version]-x86_64-linux-bin.tar.gz

Now you need to know which GPU architecture you're targeting. This is
typically an ``sm``\_ prefix followed by a number. For example
``sm_89`` is the Ada Lovelace architecture. You can find details `on
the GPU architecture mapping here
<https://arnon.dk/matching-sm-architectures-arch-and-gencode-for-various-nvidia-cards/>`_.
You pass this parameter to the next script.

In the extracted directory, generate the tool suite setup for your
current installation:

.. code-block:: shell

  cd cuda_env-[version]-x86_64-linux-bin/cuda
  sh setup.sh -mcpu <your GPU architecture>

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

You need only perform this step at installation. You should see:

.. code-block:: shell

  CUDA kernel launch with  16 blocks of  256  threads
  Copy output data from the CUDA device to the host memory
  Test PASSED
  Done
