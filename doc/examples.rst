**************************************
Examples
**************************************

Vector Add
==========

TODO

Marching Cubes
==============

Marching cubes is one of the typical algorithms in graphical rendering. It 
allows to convert a density function able to separate absence of presence of
a material in a continuous 3D space into a mesh of triangles. This algorithm
is a transcription of the algoritm details in NVIDIA's `Metaballs GPU Gem 3 manual 
<https://developer.nvidia.com/gpugems/gpugems3/part-i-geometry/chapter-1-generating-complex-procedural-terrains-using-gpu>`_.
In this example, we'll define a density function through `Metaballs <https://en.wikipedia.org/wiki/Metaballs>`_

.. image:: marching.png

To be able to build and run the example, make sure that you have on your system 
the following dependencies installed:

 - SDL
 - OpenGL

Building the example should be similiar to building vectorAdd in the 
installation step. Remember that if not already done, you need to have
sourced env.sh in your environment first::

 cd <your gnat cuda installation>
 . env.sh
 cd cuda/examples/marching
 make
 ./main

TODO: Give instructions on how to set the right SM!!!

This should open a window and display metaballs on the screen moving around.
Depending on the GPU power available on your system, you may have a more
or less fast rendering. This can be adjusted by changing the sampling of the 
grid that computes marching cubes - the smaller the sampling the faster the 
computation. You can adust that by changing the value under ``src/common/data.ads``::

    Samples : constant Integer := 256;

Try for example 128 or 64. This value needs to be a power of 2.

TODO