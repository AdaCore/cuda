**************************************
Performance Considerations
**************************************

GNAT for CUDA relies on the backend technology from the CUDA toolchain, and
offers a very similar compilation model. There is no fundamental reason why 
a program in Ada would be significantely different than a program in C from
a performance and resource consumption point of view. However, usage of some
capabilities or pattern may alter performances.

Compiler Switches
=================

The following switches can be considered:

- ``-gnatp`` will remove checks automatically added by the Ada language, and will
  make the application run faster.
- at least ``-O2`` optimization levels is recommended
- ``-gnatn`` will enable inlining.

Unconstrained Arrays
====================

In Ada, unconstrained arrays are associated with two bounds, which are not known
statically. This differs from their C equivalent which always start at zero.
The consequence is that computing the offset of an array requires a memory read.
E.g.:

.. code-block:: ada

       type Arr is array (Integer range <>) of Integer;
       type Arr_Access is access all Arr;
       V : Arr_Access := new Arr (5 .. 10);
    begin
       V (6) := 0;

When computing the memory location of item 6 in the array, the code must furst
load from memory the lowerbound (5), then substract the index to this lower 
bound to compute the memory offset from the array starting point.

To solve this issue, an Ada extension is available when compiling with ``-gnatX``.
This allows to fix the lower bound of the array. If this lower bound is zero,
it will remove the need for both loading that lower bound and adding a value
when computing the offset. E.g.:

.. code-block:: ada

       type Arr is array (Integer range 0 .. <>) of Integer;
       type Arr_Access is access all Arr;
       V : Arr_Access := new Arr (0 .. 5);
    begin
       V (1) := 0;

The feature is fully described `here <https://github.com/AdaCore/ada-spark-rfcs/blob/master/considered/rfc-lower-bound.rst>`_.