***********************************
Prerequisites and environment setup
***********************************

Before installing this software, make sure you have the pre-requisites
corresponding to your build environment installed. 
CUDA Toolkit and a suitable GNAT Ada compiler are required for building the
CUDA bindings.

.. note:: 
  
  In cross compilation workflow the version of CUDA Toolkit on the development
  host must be the same as that on the target.

.. _DEFAULT_INSTALL:

CUDA Toolkit installation on a workstation with NVIDIA GPU
**********************************************************

In case the machine used for development has a CUDA-capable NVIDIA GPU the toolkit
can be installed by following the standard setup instructions from NVIDIA.
Start from downloading the  CUDA Toolkit for your development host from 
https://developer.nvidia.com/cuda-downloads.

You need to have the CUDA Toolkit in your PATH, and in particular ``ptxas``.
You can check that by running:

.. code-block:: shell

  which ptxas

If it doesn't return anything, CUDA may not be properly installed,
or needs to be put in your PATH, e.g.:

.. code-block:: shell

   export PATH=/usr/local/cuda/bin:$PATH

.. _CUSTOM_INSTALL:

CUDA Toolkit installation on a workstation without a suitable GPU
*****************************************************************

In case the development host doesn't have CUDA-capable GPU, the available GPU
is not compliant with that on the target or the development environment needs
to be installed without root permissions, the toolkit can be installed without
video card drivers.

Downloading the CUDA Toolkit in **runfile format** for your development host from 
https://developer.nvidia.com/cuda-downloads.

Decide where the toolkit shall be installed and expose the location of the toolkit
with environment variable ``CUDA_ROOT``.

.. warning::

   ``CUDA_ROOT`` cannot point to a folder that contains a ``gcc`` or ``gnat`` installation in any of its subdirectories.
   By default, ``gcc`` is installed in :file:`/usr`. Avoid installing a custom CUDA toolkit in the same folder.

.. code-block:: shell

  mkdir cuda-toolkit
  export CUDA_ROOT=`pwd`/cuda-toolkit

Install the toolkit using the runfile downloaded from NVIDIA website using the
options listed below:

.. code-block:: shell

  sh cuda_<cuda version>_linux.run --silent --toolkit --toolkitpath=$CUDA_ROOT --override --defaultroot=$CUDA_ROOT/root

Expose CUDA libraries for the linker and binaries for the setup script:

.. code-block:: shell

  export LD_LIBRARY_PATH=$CUDA_ROOT/targets/<architecture>/lib:$LD_LIBRARY_PATH
  export PATH=$CUDA_ROOT/bin:$PATH

<architecture> above is the name of the architecture for the target platform.

Native compiler on x86_64 Linux
*******************************

In case both the development host and the target are running x86_64 Linux
then the following tools are required:

 - An x86_64 Linux environment with CUDA drivers (see above)
 - An installation of GNAT Pro, version 24.0w (20230413) or later.

Cross compilation for aarch64 Linux
***********************************

If the development host is running x86_64 Linux and the target
aarch64 Linux then the following tools are required:

 - An aarch64 Linux environment with CUDA drivers on the target.
 - An installation of GNAT Pro cross toolchain for aarch64-linux, 
   version 24.0w (20230413) or later, on the development host.

Obtain a copy of the system libraries according to the instructions 
in the cross toolchain documentation and place them in a directory of
your choice. **NB!** if you are going to copy the folders from target
to the development host then make sure that all of the required
libraries are installed on target before.

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

  $ scp -rp <my-aarch64-linux-target>:/usr/local/cuda/targets/aarch64-linux ./
  $ sudo mv aarch64-linux <CUDA_TOOLBOX_ROOT>/targets

Where <CUDA_TOOLBOX_ROOT> is the location of the cuda toolbox:

* ``$CUDA_ROOT`` in case the toolbox was installed according to the instructions
  in :ref:`CUSTOM_INSTALL`

* ``/usr/local/cuda`` in case of :ref:`DEFAULT_INSTALL`

Make the sysroot location visible to GNAT via the ``ENV_PREFIX`` environment 
variable:

.. code-block:: shell

  $ export ENV_PREFIX=`pwd`/sysroot

Let the toolchain know that the intended compilation target is aarch64-linux:

.. code-block:: shell

  $ export CUDA_HOST=aarch64-linux

***************
GNAT-CUDA setup
***************

After setting up the environment, you can extract the gnat-cuda package:

.. code-block:: shell

   tar -xzf gnat-cuda-[version]-x86_64-linux-bin.tar.gz

Now you need to know which GPU architecture you're targeting. This is
typically an ``sm``\_ prefix followed by a number. For example
``sm_89`` is the Ada Lovelace architecture. You can find details from
the `GPU architecture mapping article
<https://arnon.dk/matching-sm-architectures-arch-and-gencode-for-various-nvidia-cards/>`_.
You pass this parameter to the next script.

In the extracted directory, generate the tool suite setup for your
current installation:

.. code-block:: shell

  cd gnat-cuda-[version]-x86_64-linux-bin/cuda
  ./setup.sh [-mcpu sm_<GPU architecture>] [-clean]

If the ``-mcpu`` argument is not provided, then the setup attempts to determine
the compute capability automatically using the utilities in CUDA toolbox.

The ``-clean`` argument can be optionally used for removing the temporary object
files in case the environment changes and the change cannot be detected automatically 
by the binding generation process. This can happen, for instance, when the compiler
is upgraded or the same gnat-cuda source tree is used for multiple targets 
(e.g. for native x86_64-linux build and aarch64-linux cross compilation) and 
you switch from one target to another by changing the value of `$CUDA_HOST`
variable.

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

  In cross compilation workflow you have to copy ``main`` to target
  before executing it.

.. note::

  If you are switching between different targets by changing the
  ``$CUDA_HOST`` variable or upgraded the compiler then the old 
  object files can be removed by calling ``make clean`` before
  a new build.

After executing the code you should see:

.. code-block:: shell

  CUDA kernel launch with  16 blocks of  256  threads
  Copy output data from the CUDA device to the host memory
  Test PASSED
  Done
