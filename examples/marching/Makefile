build:
	Execution_Side=Device gprbuild -P ../../api/cuda_device.gpr -j0
	Execution_Side=Device gprbuild -P marching_device_code.gpr -j0
	Execution_Side=Host gprbuild -P marching_cubes -j0 -largs $(PWD)/obj/device/*.fatbin.o
