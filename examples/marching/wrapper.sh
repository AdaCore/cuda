#!/bin/sh -e

KERNEL_SRC_NAME="marching_cubes.adb"

Execution_Side=Device gprbuild -P../../api/cuda
Execution_Side=Device gprbuild --config=cuda.cgpr -c -P marching_cubes.gpr $KERNEL_SRC_NAME -v
Execution_Side=Host gprbuild -Pmarching_cubes -largs $PWD/obj/*.fatbin.o
