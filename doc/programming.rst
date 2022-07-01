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

Using Storage Model Library
---------------------------

Note - this method is experimental and is provided to bridge the gap pending 
implementation of the storage model aspect described later.

One of the most useful thing to do in CUDA is to pass arrays back and forth
and to compute values on them. Unfortunately, an Ada array is more complex than
a C array and cannot be allocated using a simple malloc invocation. Notably,
Ada array (or more specifically Ada unconstrained arrays) carry data and 
boundaries. The structure of such types in memory is implementation-dependent,
and can vary on many factors.

GNAT for CUDA currently provides a storage model library that allows to allocate
uni-dimensional arrays and to copy them back and forth easily. This is done
through the generic package ``CUDA_Storage_Models.Malloc_Host_Storage_Model.Arrays``
which can be instantiated with for generic formal parameters:

.. code-block:: ada

   type Typ is private; -- the type of component
   type Index_Typ is (<>); -- the type of indexes
   type Array_Typ is array (Index_Typ range <>) of Typ; -- the array type
   type Array_Access is access all Array_Typ; -- a pointer type to the array

For example:

.. code-block:: ada

   type Int_Array is array (Integer range <>) of Integer;
   type Int_Array_Access is access all Int_Array;

   package Int_Device_Arrays is new CUDA_Storage_Models.Malloc_Storage_Model.Arrays 
    (Integer, Integer, Int_Array, Int_Array_Access);

Once instantiated, the newly created package exports a type ``Foreign_Access``
which designates a handle to the array in device memory, together with 
allocation, assignment and deallocation functions:

.. code-block:: ada

   type Foreign_Array_Access is record
      Data   : Foreign_Address;
      Bounds : Foreign_Address;
   end record;

   function Allocate (First, Last : Index_Typ) return Foreign_Array_Access;
   function Allocate_And_Init (Src : Array_Typ) return Foreign_Array_Access;

   procedure Assign
     (Dst : Foreign_Array_Access; Src : Array_Typ);
   procedure Assign
     (Dst : Foreign_Array_Access; First, Last : Index_Typ; Src : Array_Typ);
   procedure Assign
     (Dst : Foreign_Array_Access; Src : Typ);
   procedure Assign
     (Dst : Foreign_Array_Access; First, Last : Index_Typ; Src : Typ);
   procedure Assign
     (Dst : in out Array_Typ; Src : Foreign_Array_Access);
   procedure Assign
     (Dst : in out Array_Typ; Src : Foreign_Array_Access; First, Last : Index_Typ);

   procedure Deallocate (Src : in out Foreign_Array_Access);

Note that the above declaration is a simplification of the full package.

This can then be used to allocate memory, and perform back and forth copies from
host to device:

.. code-block:: ada

    procedure Main is
       Host_Array : Int_Array_Access := new Int_Array (1 .. 100);
       Device_Array : Int_Device_Arrays.Foreign_Access;
    begin
       Host_Array.all := (others => 0);
       Device_Array := Allocate (1, 100);

       Assign (Device_Array, Host_Array.all)
       
       pragma Kernel_Execute (
           Some_Kernel (Uncheck_Convert (Device_Array)),
           Host_Array.all'Length,
           1);

       Assign (Host_Array.all, Device_Array)
    end Main;

Note the call of ``Uncheck_Convert`` when calling the kernel. This function is 
declared as such:

.. code-block:: ada

    function Uncheck_Convert (Src : Foreign_Access) return Typ_Access;

It allows to convert a ``Foreign_Access`` to a regular access to array. However, the
memory accessed by this pointer is located on the device, not the host, so any
direct access from the host will lead to memory errors.

The device code can now rely on an actual array access:

.. code-block:: ada

    procedure Kernel (Device_Array : Int_Array_Access) is
    begin
       Device_Array (Thread_IDx.X) := Device_Array (Thread_IDx.X) + 10;
    end Kernel;

While this is clearly an improvement over the low level data transfer method, 
this is clearly not satisfactory. Notably, the ``Uncheck_Convert`` creates an
object that looks usable from the host, but which usage there will lead to memory
errors.

Using Storage Model Aspect
--------------------------

Storage Model is an extension to the Ada language that is currently under 
implementation. It is not yet available as part of the current version of the 
product but is on the close roadmap. Discussion around the generic capability 
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
