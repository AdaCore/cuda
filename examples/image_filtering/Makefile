build:
	Execution_Side=Device gprbuild -P ../../api/cuda_device.gpr -j0 -v
	Execution_Side=Device gprbuild -P device.gpr -j0 -v
	Execution_Side=Host gprbuild -P host -j0 -largs $(PWD)/obj/device/*.fatbin.o -v
