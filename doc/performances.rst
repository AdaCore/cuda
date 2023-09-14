**************************************
Performance Considerations
**************************************

.. role:: switch(samp)

GNAT for CUDA relies on the CUDA toolchain backend technology and provides
a very similar compilation model. There's no fundamental reason why the
performance and resource consumption of a program in Ada would be
essentially different than that of the same program in C. However, you
may need to specify some non-default switches to get the best performance
and your usage of some Ada features may negatively affect performance.

Compiler Switches
=================

You should consider using the following switches:

- :switch:`-gnatp` removes checks that are added by the Ada language by
  default.  Removing those checks makes the application run faster.
- we recommend at least the :file:`-O2` optimization level
- :switch:`-gnatn` enables inlining.

Unconstrained Arrays
====================

In Ada, unconstrained arrays are associated with two bounds per dimension,
which are not known statically. This differs from their C equivalent where
the index always starts at zero and which have known bounds.  The
consequence is that computing the offset of an Ada array normally requires
a memory read.  E.g.:

.. code-block:: ada

       type Arr is array (Integer range <>) of Integer;
       type Arr_Access is access all Arr;
       V : Arr_Access := new Arr (5 .. 10);
       begin
          V (6) := 0;

When computing the memory location of item 6 in the array, the code must
first load the lower bound (5) from memory and then subtract this lower
bound from the index to compute the memory offset from the start of the
array. If you do this in a loop over the array, the compiler can move
this operation out of the loop, making it considerably less expensive, but
if you do only a single array operation, the computation of the starting
position is significant.

To improve this performance, an Ada extension is available when compiling
with :switch:`-gnatX0`. This extension indicates that the lower bound of
the array is known at compile-time. If this lower bound is a constant, it
will remove both the need for loading that lower bound and doing a
subtraction when computing the offset. E.g.:

.. code-block:: ada

       type Arr is array (Integer range 0 .. <>) of Integer;
       type Arr_Access is access all Arr;
       V : Arr_Access := new Arr (0 .. 5);
       begin
          V (1) := 0;


.. only:: COMMENT

  Such a file doesn't exist in ada-spark-rfcs repository

The feature is fully described `here
<https://github.com/AdaCore/ada-spark-rfcs/blob/master/considered/rfc-lower-bound.rst>`_.
