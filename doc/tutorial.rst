**************************************
Tutorial
**************************************

Some Expensive Computation
==========================

Before compiling any CUDA code, you will need to source :file:`env.sh` into
your environment::

Open the :file:`tutorial` directory found at the top directory of this CUDA
repository. There you'll see a typical CUDA project structure. In
particular, :file:`device.gpr` builds the code that runs on the device and
:file:`host.gpr` builds the code for the host.  :file:`Makefile` is
responsible for building both projects.  Note that it's using the standard
:file:`Makefile.build` structure. If you look at
:file:`cuda/Makefile.build`, you'll see both build commands:

.. code-block:: Makefile

    gprbuild -Xcuda_host=$(CUDA_HOST) -P device
    gprbuild -Xcuda_host=$(CUDA_HOST) -P host -largs $(CURDIR)/lib/*.fatbin.o

The switch :switch:`-Xcuda_host=$(CUDA_HOST)` allows building for cross
platform such as ARM Linux. In this tutorial, we're only building for
native X86 Linux so it isn't needed.

An important switch on the host compilation command line is :switch:`-largs
$(CURDIR)/lib/*.fatbin.o`.  This adds the object code containing the device
binary to the list of object files linked with the host application.  The
host will load this library onto the GPU.

You can make and run the application:

.. code-block:: shell

    make && ./main

You should see timing output, e.g.:

.. code-block:: text

    Host processing took  0.021741000 seconds
    Device processing took  0.000000000 seconds

This sample project is doing expensive computations with O(3) complexity on
an array whose size is 2^8 by default. We haven't yet arranged for any
processing to be done on the device, which explains why there was no time
spent on the device. You can change the size of the array by adding a
parameter on the :file:`main` invocation, e.g.:

.. code-block::

    ./main 9
    Host processing took  1.303823000 seconds
    Device processing took  0.000000000 seconds

The timing depends on your platform. You should chose a parameter that
causes the processing to take around 1 second to make the next steps
meaningful.

As we said, this code is currently fully native and single treaded. We're
now going to offload the computation to the GPU.

Open :file:`src/common/kernel.ads`. You'll see the specification of
:code:`Complex_Computation`:

.. code-block:: ada

  procedure Complex_Computation
     (A : Float_Array;
      B : Float_Array;
      C : out Float_Array;
      I : Integer);

We're going to wrap this call into a CUDA kernel that we can call from the
host.

We first need to create types. Ada arrays can't be passed directly from
host to device: they need to be passed through specific access types marked
as addressing device memory space. You do this by using a specific aspect
on the type :code:`Designated_Storage_Model => CUDA.Storage_Models.Model`.
When you do this, allocation and deallocation are done through the CUDA
API.  Copies betwen these pointers and native pointers are also modified to
move data from the device to the host and back.

We next introduce a new pointer type in the :code:`Kernel` package:

.. code-block:: ada

    type Array_Device_Access is access Float_Array
       with Designated_Storage_Model => CUDA.Storage_Models.Model;

This pointer must be pool specific: it can't be an :code:`access all` type.
That means it conceptually points to a specific pool of data, the device
memory, and that conversions between other pointers types aren't allowed.

We're now going to introduce a procedure to be called from the host. In the
CUDA world, this is called a "kernel". Kernels are identified by a special
aspect, :code:`CUDA_Global` which corresponds to the :code:`__global__`
modifier used in C CUDA code. This kernel accepts :code:`A`, :code:`B` and
:code:`C` as parameters. The specific index, :code:`J`, isn't passed to the
kernel, but is instead computed there.

Write the kernel specification:

.. code-block:: ada

   procedure Device_Complex_Computation
     (A : Array_Device_Access;
      B : Array_Device_Access;
      C : Array_Device_Access)
     with CUDA_Global;

We're now going to implement the kernel. Open :file:`kernel.adb` and start
writing the body of the function:

.. code-block:: ada

   procedure Device_Complex_Computation
     (A : Array_Device_Access;
      B : Array_Device_Access;
      C : Array_Device_Access)
    is
    begin
       null;
    end Device_Complex_Computation;

This kernel is called in parallel, once per index in the array we're
computing. Within a kernel, you can index a given call using the thread
number (:code:`Thread_IDx`) and the block number (:code:`Block_IDx`). You
can also retrieve the number of threads in a block that have been scheduled
(:code:`Block_Dim`) and the number of blocks in the grid
(:code:`Grid_Dim`). These correspond to three dimension values, which we
can call :code:`x`, :code:`y`, and :code:`z`. In this example, we're only
going to use the :code:`x` dimension.

Next, we add a computation of the index, :code:`J`, into the body of the
kernel based on the block and thread index:

.. code-block:: ada

   J : Integer := Integer (Block_Dim.X * Block_IDx.X + Thread_IDx.X);

These are expressed in terms of :code:`Interfaces.C.int`, so we need to 
explicly convert the result to :code:`Integer`.

At this point the call to :code:`Complex_Computation` is trivial. Our whole
kernel should now look like:

.. code-block:: ada

   procedure Device_Complex_Computation
     (A : Array_Device_Access;
      B : Array_Device_Access;
      C : Array_Device_Access)
   is
      J : Integer := Integer (Block_Dim.X * Block_IDx.X + Thread_IDx.X);
   begin
      Complex_Computation (A.all, B.all, C.all, J);
   end Device_Complex_Computation;

We're now done with the kernel, so let's move on to the host code. Open
:file:`src/host/main.adb`.  That code currently allocates three native
arrays, :code:`H_A`, :code:`H_B` and :code:`H_C` and runs a computation on
them. We're going to introduce three new arrays, :code:`D_A`, :code:`D_B`,
and :code:`D_C` using the :code:`Array_Device_Access` type we created
earlier.

The new declaration is:

.. code-block:: ada

    D_A, D_B, D_C : Array_Device_Access;

We're now going to determine how kernel calls will be scheduled on the GPU.
In this tutorial, we're going to have each block contain 256 threads.  We
can compute the number of blocks to cover the entire array, which is
essentially :code:`Number_Of_Elements / Threads_Per_Block`, but we add 1 to
account for leftover portions of :code:`Threads_Per_Block`.

The computations for :code:`Threads_Per_Block` and :code:`Blocks_Per_Grid`
are:

.. code-block:: ada

   Threads_Per_Block : Integer := 256;
   Blocks_Per_Grid   : Integer := Num_Elements / Threads_Per_Block + 1;

We now need to allocate memory on the device side. To compute the actual
additional cost of device computation, this allocation is taken into
account in the total time reported because data copy can be a critically
limiting factor of GPU performance enhancements.

Find the portion of the body marked :code:`-- INSERT HERE DEVICE
CALL`. After that, add the two array allocations and copies for :code:`H_A`
and :code:`H_B` to :code:`D_A` and :code:`D_B` respectively.  Also allocate
an array for :code:`D_C` which is the size of :code:`H_C`:

.. code-block:: ada

   D_A := new Float_Array'(H_A.all);
   D_B := new Float_Array'(H_B.all);
   D_C := new Float_Array (H_C.all'Range);

These three statements are using the storage model introduced before. In
particular, allocations are done through the CUDA API and copies from the
host to the device are also done through the CUDA API.

Now we can finally call our kernel code! We do this using a special pragma,
:code:`CUDA_Execute`, which takes at least three parameters: a procedure
call to a kernel, the dimension of the blocks (how many threads they
contain) and the dimension for the grid (how many block it contains).

The CUDA call is as follows:

.. code-block:: ada

   pragma CUDA_Execute
     (Device_Complex_Computation (D_A, D_B, D_C),
      Threads_Per_Block,
      Blocks_Per_Grid);

When executing that pragma, the CUDA API schedules
:code:`Device_Complex_Computation` to be executed :code:`Blocks_Per_Grid *
Threads_Per_Block` times on the kernel. This call itself is non-blocking,
but subsequent dependent operations (such as copies from the device) will
block host execution until the kernel completes.

Let's introduce this copy now. Results are going to be stored in
:code:`D_C`, so let's copy it to :code:`H_C`:

.. code-block:: ada

    H_C.all := D_C.all;

This is a copy between a host and a device pointer, which will be
implemented as a copy from device memory to the host.

The whole sequence should look like:

.. code-block:: ada

   D_A := new Float_Array'(H_A.all);
   D_B := new Float_Array'(H_B.all);
   D_C := new Float_Array (H_C.all'Range);

   pragma CUDA_Execute
     (Device_Complex_Computation (D_A, D_B, D_C),
      Threads_Per_Block,
      Blocks_Per_Grid);

   H_C.all := D_C.all;

That's it! As an extra exercise, you might want to instantiate and call
:code:`Ada.Unchecked_Deallocation` on the device pointers, but that's not
strictly necessary. Now compile and run the code. You can try different
values for the array size to observe different timings. For example:

.. code-block::

    ./main 10
    Host processing took  1.227895000 seconds
    Device processing took  0.051606000 seconds

Marching Cubes
==============

The marching cubes example demonstrates a more interesting
computation. Marching cubes is an important algorithms in graphical
rendering. It converts a density function, which indicates the absence or
presence of a material in a continuous 3D space, into a mesh of
triangles. The algorithm in this example is a transcription of the
algorithm shown in NVIDIA's `Metaballs GPU Gem 3 manual
<https://developer.nvidia.com/gpugems/gpugems3/part-i-geometry/chapter-1-generating-complex-procedural-terrains-using-gpu>`_.
In this example, we'll define a density function through `Metaballs
<https://en.wikipedia.org/wiki/Metaballs>`_

.. image:: marching.png

To build and run the example, ensure you have SDL and OpenGL installed.
You can build and run the code like the other examples:

.. code-block::

    cd cuda/examples/marching
    make
    ./main

This opens a window and shows metaballs moving around the screen.  The
speed of the rendering is dependent on the GPU power available on your
system.  You can adjust the speed by changing the sampling of the grid that
computes the marching cubes: the smaller the sampling, the faster the
computation.  You do this by changing the value in
:file:`src/common/data.ads`::

.. code-block:: ada

    Samples : constant Integer := 256;

This value needs to be a power of 2.  Try 128 or 64, for example.

A detailed walkthrough of this code is beyond the scope of this tutorial,
but this example is a good place to start looking at more complex usage of
the technology.
