n**************************************
Programming with GNAT for CUDA®
**************************************

CUDA API
========

The CUDA API available from GNAT for CUDA® is a binding to the CUDA API
provided by NVIDIA.  The NVIDIA API is installed with the CUDA driver.  You
access the Ada API by adding a reference to :file:`cuda_host.gpr` (on the
host) and :file:`cuda_device.gpr` (on the target).  The initial
installation script generates the Ada version of the API from the CUDA
version that's installed on your system.

Two versions of the Ada API are available:

- a "thick" binding version. These are child units of the :code:`CUDA`
  package, the main one being :code:`CUDA.Runtime_API`. This is the API you
  will most likely use. However, this API is still in the process of being
  completed and a number of types and subprogram specs have not been mapped
  to higher-level Ada constructs. For example, you will still see a lot of
  references to :code:`System.Address` where you would normally expect
  specific access types in Ada.
- a "thin" binding version. These are typically identified by having a
  suffix of :code:`_h`.  They are direct bindings to the underlying C
  APIs. These bindings are functional and complete.  They can be used as a
  low-level alternative to the thick binding, but they don't expose an
  interface consistent with the Ada programming style and may require more
  work to use.

You can regenerate these bindings at any time. You may want to do this, for
example, if you install a new version of CUDA. To regenerate these
bindings, execute the :file:`bind.sh` script located in :file:`<your GNAT
for CUDA installation>/cuda/api/`.

Defining and calling Kernels
============================

Just as in a typical CUDA program, programming in GNAT for CUDA requires
you to identify application entry points to the GPU code, called
"kernels". In Ada, you do this by annotating a procedure with the
:code:`CUDA_Global` aspect, which serves the same role as the CUDA
:code:`__global__` modifier. For example:

.. code-block:: ada

    procedure My_Kernel (X : Some_Array_Access)
    with CUDA_Global;

Kernels are compiled both for the host and the device. They can be called
as regular procedures, e.g:

.. code-block:: ada

    My_Kernel (An_Array_Instance);

The above makes a regular single-threaded call to the kernel and executes
it on the host.  You may want to do this because of a better debugging
environment on the host.

To call a kernel on the device (which means copying it to the device and
executing it there), you use the :code:`CUDA_Execute` pragma:

.. code-block:: ada

    pragma CUDA_Execute (My_Kernel (An_Array_Instance), 10, 1);

The procedure call looks the same as a regular call, but this call is
surrounded by the pragma :code:`CUDA_Execute`, which has two extra
parameters defining, respectively, the number of threads per block and the
number of blocks per grid. This is equivalent to the CUDA call:

.. code-block:: c

    <<<10, 1>>> myKernel (someArray);

In each case, these calls launch ten instances of the kernel to the device.

The numbers of threads per block and blocks per grid can be expressed as a
one-dimensional scalar or a :code:`Dim3` value that specifies all three
dimensions (:code:`x`, :code:`y`, and :code:`z`). For example::

.. code-block:: ada

   pragma CUDA_Execute (My_Kernel (An_Array_Instance), (3, 3, 3), (3, 3, 3));

The above call launches (3 * 3 * 3) * (3 * 3 * 3) = 729 instances of the
kernel on the device.

Passing Data between Device and Host
====================================

Using Storage Model Aspect
--------------------------

"Storage Model" is an extension to the Ada language that is currently under
development. General discussions about the capability can be found `here
<https://github.com/AdaCore/ada-spark-rfcs/pull/76>`_.

GNAT for CUDA provides a storage model that maps to CUDA primitives for
allocation, deallocation, and copying. The model is declared in the package
:code:`CUDA.Storage_Models`.  You may either use
:code:`CUDA.Storage_Models.Model` itself or you may create your own.

When a pointer type is associated with a CUDA storage model, memory
allocation through that pointer occurs on the device in the same manner as
it would in the host if a storage model wasn't specified.  For example:

.. code-block:: ada

    type Int_Array is array (Integer range <>) of Integer;

    type Int_Array_Device_Access is access Int_Array
       with Designated_Storage_Model => CUDA.Storage_Model.Model;

    Device_Array : Int_Array_Device_Access := new Int_Array (1 .. 100);    

In addition to allocation being done on the device, copies between the host
and device are convverted to call the CUDA memory copy operations. So you
can write:

.. code-block:: ada

    procedure Main is
       type Int_Array_Host_Access is access Int_Array;

       Host_Array : Int_Array_Host_Access := new Int_Array (1 .. 100);
       Device_Array : Int_Array_Device_Access := new Int_Array'(Host_Array.all);
    begin
       pragma Kernel_Execute (
           Some_Kernel (Device_Array),
           Host_Array.all'Length,
           1);

       Host_Array.all := Device_Array.all;
    end Main;

On the kernel side, :code:`CUDA.Storage_Model.Model` is the native storage
model (as opposed to the foreign device one when on the host side).  You
can use :code:`Int_Array_Device_Access` directly:

.. code-block:: ada

    procedure Kernel (Device_Array : Int_Array_Device_Access) is
    begin
       Device_Array (Thread_IDx.X) := Device_Array (Thread_IDx.X) + 10;
    end Kernel;

This is the recommended way of sharing memory between device and host.
However, the storage model can be extended to support capabilities such as
streaming or unified memory.

Using Unified Storage Model
---------------------------

An alternative to using the default CUDA Storage model is to use so-called
"unified memory". In that model, the device memory is mapped directly onto
host memory, so no special copy operation is necessary. The factors that
may lead you to choose to one model over the other are outside of the scope
of this manual. To use unified memory, you use the package
:code:`Unified_Model` instead of the default one:

.. code-block:: ada

    type Int_Array is array (Integer range <>) of Integer;

    type Int_Array_Device_Access is access Int_Array
       with Designated_Storage_Model => CUDA.Storage_Model.Unified_Model;

Using Storage Model with Streams
--------------------------------

CUDA streams allows you to launch several computations in parallel. This
model allows you to specify which computation write and read operation must
wait for. The Ada CUDA API doesn't provide a pre-allocated stream memory
model. Instead, it provides a type, :code:`CUDA_Async_Storage_Model`, that
you can instantiate and specify the specific stream::

.. code-block:: ada

    My_Stream_Model : CUDA.Storage_Model.CUDA_Async_Storage_Model
      (Stream => Stream_Create);

    type Int_Array is array (Integer range <>) of Integer;

    type Int_Array_Device_Access is access Int_Array
       with Designated_Storage_Model => My_Stream_Model;

The data stream associated with a specific model can vary over time,
allowing different parts of a given object to be used by different streams,
e.g.:

.. code-block:: ada

       X : Int_Array_Device_Access := new Int_Array (1 .. 10_000);
       Stream_1 : Stream_T := Stream_Create;
       Stream_2 : Stream_T := Stream_Create;
    begin
       My_Stream_Model.Stream := Stream_1;
       X (1 .. 5_000) := 0;
       My_Stream_Model.Stream := Stream_2;
       X (5_001 .. 10_000) := 0;

Low-Level Data Transfer
-----------------------

At the lowest level, you can allocate memory to the device using the
standard CUDA function :code:`malloc` that's bound from
:code:`CUDA.Runtime_API.Malloc`. E.g.:

.. code-block:: ada

 Device_Array : System.Address := CUDA.Runtime_API.Malloc (Integer'Size * 100);

This is equivalent to the following CUDA code:

.. code-block:: c

 int *deviceArray = cudaMalloc (sizeof (int) * 100);

In this example, objects on the Ada side aren't typed. Creating typed
objects requires more advanced Ada constructions that are described later.

The above statement created space in the device memory of 100 integers.
That space can now be used to perform copies back and forth from host
memory. For example:

.. code-block:: ada

    procedure Main is
       type Int_Array is array (Integer range <>) of Integer;
       type Int_Array_Access is access all Int_Array;

       Host_Array : Int_Array_Access := new Int_Array (1 .. 100);
       Device_Array : System.Address := CUDA.Runtime_API.Malloc (Integer'Size * 100);
    begin
       Host_Array := (others => 0);

       CUDA.Runtime_API.Memcpy
           (Dst   => Device_Array,
            Src   => Host_Array.all'Address,
            Count => Host_Array.all'Size,
            Kind  => Memcpy_Host_To_Device);

        pragma Kernel_Execute (
            Some_Kernel (Device_Array, Host_Array.all'Length),
            Host_Array.all'Length,
            1);

        CUDA.Runtime_API.Memcpy
           (Dst   => Host_Array.all'Address
            Src   => Device_Array,
            Count => Host_Array.all'Size,
            Kind  => Memcpy_Device_To_Host);
    end Main;

This code copies the contents of :code:`Host_Array` to
:code:`Device_Array`, performs some computations on that data on the
device, and then copies the data back. At this level of coding, we're not
passing a typed array but instead a raw address. On the kernel side, we
need to reconstruct the array with an overlay:

.. code-block:: ada

    procedure Kernel (Array_Address : System.Address; Length : Integer) is
       Device_Array : Int_Array (1 .. Length)
          with Address => Array_Address;
    begin
       Device_Array (Thread_IDx.X) := Device_Array (Thread_IDx.X) + 10;
    end Kernel;

While it works, this method of passing data back and forth is not very
satisfactory and you should reserve it for cases where an alternative
doesn't exist or doesn't exist yet. In particular, typing is lost at the
interface, and you need to carefully check manually for type correctness.

Specifying Where Code is For
============================

Like in CUDA, a GNAT for CUDA application contains code that may be
compiled exclusively for the host, the device, or both. By default, all
code is compiled for both the host and the device. You can identify code as
only being compilable for the device by using the :code:`CUDA_Device`
aspect:

.. code-block:: ada

   procedure Some_Device_Procedure
      with CUDA_Device;

:code:`Some_Device_Procedure` will not exist on the host. Calling it will
result in a compilation error.

The corresponding :code:`CUDA_Host` aspect is currently not implemented.

Accessing Block and Thread Indexes and Dimensions
=================================================

GNAT for CUDA® allows you to access block and thread indexes and
dimensions in a way that's similar to CUDA. The package
:code:`CUDA.Runtime_API` declares :code:`Block_Dim`, :code:`Grid_Dim`,
:code:`Block_IDx` and :code:`Thread_IDx` which map directly to the
corresponding PTX registers. For example:

.. code-block:: ada

    J : Integer := Integer (Block_Dim.X * Block_IDx.Y + Thread_IDx.X);
