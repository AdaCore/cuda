**************************************
Programming with GNAT for CUDA®
**************************************

CUDA API
========

The CUDA API available from GNAT for CUDA® is a binding to the CUDA API 
provided by NVIDIA, installed with the CUDA driver. Is is accessed by the host
by adding a reference to ``cuda_host.gpr`` on the host and ``cuda_device.gpr``
on the target.

The Ada version of the API is generated automatically when running the initial
installation script, and thus corresponds specifically to the CUDA version that
is installed on the system.

Two version of the API are available:

- a "thick" binding version. These units are child units of the CUDA package,
  the main one being ``CUDA.Runtime_API``. This is the intended API to use.
  Note that at this stage, this API is still in process of being completed.
  A number of types and subprogram profiles have not been mapped to higher
  level Ada constructions. For example, you will still see a lot of references
  to ``System.Address`` where Ada would call for specific types.
- a "thin" binding version. These units are typically identified by their 
  suffix "_h" and are direct bindings to the underlying C APIs. These bindings
  are functional and complete, they can be used as a low level alternative
  to the thick binding. However, they do not expose an interface consistent 
  with the Ada programming patterns and may require more work at the user level.

At any time, these bindings can be regenerated. That can be useful for example
if a new version of CUDA is installed. To generate these bindings, you can 
execute the the "bind.sh" script locaed under 
<your GNAT for CUDA installation>/cuda/api/.

Defining and calling Kernels
============================

Just as a typical CUDA program, programming in GNAT for CUDA requires the 
developper to identify in its application entry point to the GPU code called
kernels. In Ada, this is done by associating a procedure with the ``CUDA_Global``
aspect, which serves the same role as the CUDA ``__global__`` modifier. For 
example:

.. code-block:: ada

    procedure My_Kernel (X : Some_Array_Access)
    with CUDA_Global;

Kernels are compiled both for host and device. They can be called as regular
procedures, e.g:

.. code-block:: ada

    My_Kernel (An_Array_Instance);

Will do a regular single thread call to the kernel, and execute it on the host.
In some situations, this can be helpful for debugging on the host.

Calling a kernel on the device is done through the CUDA_Execute pragma:

.. code-block:: ada

    pragma CUDA_Execute (My_Kernel (An_Array_Instance), 10, 1);

Note that the procedure call looks the same than in the case of a regular call.
However, this call is done surrounded by the pragma CUDA_Execute, which has two
extra parameters, defining respectively number of threads per blocks and number
of blocks per grid. This is equivalent to a familiar CUDA call:

.. code-block:: c

    <<<10, 1>>> myKernel (someArray);

The above calls are launching ten instances of the kernel to the device.

Thread per blocks and blocks per grid can be expressed as a 1 dimention scalar
or a ``Dim3`` value which will give a dimensionality in x, y and z. For example::

.. code-block:: ada

   pragma CUDA_Execute (My_Kernel (An_Array_Instance), (3, 3, 3), (3, 3, 3));

The above call will launch (3 * 3 * 3) * (3 * 3 * 3) = 729 instances of the 
kernel on the device.

Passing Data between Device and Host
====================================

Using Storage Model Aspect
--------------------------

Storage Model is an extension to the Ada language that is currently under 
implementation. Discussion around the generic capability 
can be found `here <https://github.com/AdaCore/ada-spark-rfcs/pull/76>`_.

GNAT for CUDA provides a storage model that maps to CUDA primitives for allocation,
deallocation and copy. It is declared in the package ``CUDA.Storage_Models``.
Users may used directly ``CUDA.Storage_Models.Model`` or create their own
instances.

When a pointer type is associated with a CUDA storage model, memory allocation
will happen on the device. This allocation can be a single operation, or multiple
allocations and copies as it is the case in GNAT for unconstrained arrays. For 
example:

.. code-block:: ada

    type Int_Array is array (Integer range <>) of Integer;

    type Int_Array_Device_Access is access Int_Array
       with Designated_Storage_Model => CUDA.Storage_Model.Model;

    Device_Array : Int_Array_Device_Access := new Int_Array (1 .. 100);    

Moreover, copies between host and device will be instrumented to call proper
CUDA memory copy operations. The code can now be written:

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

On the kernel side, CUDA.Storage_Model.Model is implemented as being the native
storage model (as opposed to the foreign device one from the host). 
``Int_Array_Device_Access`` can be used directly:

.. code-block:: ada

    procedure Kernel (Device_Array : Int_Array_Device_Access) is
    begin
       Device_Array (Thread_IDx.X) := Device_Array (Thread_IDx.X) + 10;
    end Kernel;

This is the intended way of sharing memory between device and host. Note that
the storage model can be extended to support capabilities such as streaming or 
unified memory.

Using Storage Model with Streams
--------------------------------

CUDA streams allow to launch several operations in parallel. This allows to
specify which execution write and read operation have to wait for. In order
to use stream, a developer has to create a specific Storage Model

Using Unified Storage Model
---------------------------

TODO

Low Level Data Transfer
-----------------------

At the lowest level, it is possible to allocate memory to the device using the
standard CUDA function malloc bound from CUDA.Runtime_API.Malloc. E.g.:

.. code-block:: ada

 Device_Array : System.Address := CUDA.Runtime_API.Malloc (Integer'Size * 100);

This is equivalent to the following code in CUDA:

.. code-block:: c

 int * deviceArray = cudaMalloc (sizeof (int) * 100);

Note that the object on the Ada side aren't type. Creating typed objects 
requires more advanced Ada constructions that are described later.

The above example creates a space in device memory of 100 integers. It can 
now be used to perform copies back and forth from host memory. For example:

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

The above will copy the contents of Host_Array to Device_Array, perform some
computations on the device, then copy the memory back. Note that at this level
of data passing, we're not passing a typed array but a raw address. On the 
kernel side, we need to reconstruct the array with an overlay:

.. code-block:: ada

    procedure Kernel (Array_Address : System.Address; Length : Integer) is
       Device_Array : Int_Array (1 .. Length)
          with Address => Array_Address;
    begin
       Device_Array (Thread_IDx.X) := Device_Array (Thread_IDx.X) + 10;
    end Kernel;

While effective, this method of passing data back and forth is not very 
satisfactory and should be reserved for cases where not alternative exist (yet).
In particular, typing is lost at the interface, and the developer is left with
manual means of verification.


Specifying Compilation Side
===========================

As for CUDA, a GNAT for CUDA application contains code that may be compiled
exclusively for the host, the device or both. By default, all code is 
compiled for both the host and the device. Code can be identifed as only being
compilable for the device with the ``CUDA_Device`` aspect:

.. code-block:: ada

   procedure Some_Device_Procedure
      with CUDA_Device;

The above procedure will not exist on the host. Calling it will result in a
compilation error.

The correspoinding ``CUDA_Host`` aspect is currently not implemented.

Accessing Blocks and Threads Indexes and Dimensions
===================================================

GNAT for CUDA® allows to access block and thread indexes and dimensions in a way
that is similar to CUDA. Notably, the package ``CUDA.Runtime_API`` declares
``Block_Dim``, ``Grid_Dim``, ``Block_IDx`` and ``Thread_IDx`` which maps 
directly to the corresponding PTX registers. For example:

.. code-block:: ada

    I : Integer := Integer (Block_Dim.X * Block_IDx.Y + Thread_IDx.X);
