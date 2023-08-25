# matrixMul - Matrix Multiplication (CUDA Runtime API Version)

## Description
This sample demonstrates implementation of matrix multiplication in Ada.
The goal is to translate the corresponding example from NVIDIA cuda-samples repository
(https://github.com/NVIDIA/cuda-samples/tree/master/Samples/0_Introduction/matrixMul)
as close as possible. The original C code is preserved in comments to illustrate
the mapping from one language to the other.

As a notable difference, the translation doesn't implement the -device
argument. The code chooses the default GPU.

The testing algorithm fills the input matrices with a constant value.
It also contains a simple iterative computation of matrix multiplication,
which can can be used for validating the computation when a si,le constant input is replaced with random values and/or for performance comparisons.

## Usage

$ cd <sample_dir>
$ make

Run with the default settings that multiply 320x320 matrix with 640x320 matrix

$ ./main

The matrix dimensions can be changes with corresponding parameters:

$ ./main -wA=640 -hA=640 -wB=640 -hB=640

Switches -? and --help print usage information:

$ ./main -?
[Matrix Multiply Using CUDA] - Starting...
Usage [? | --help]
      [-wA=WidthA] [-hA=HeightA] (Width x Height of Matrix A)
      [-wB=WidthB] [-hB=HeightB] (Width x Height of Matrix B)
  Note: Outer matrix dimensions of A & B matrices must be equal.