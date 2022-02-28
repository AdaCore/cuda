build:
	Execution_Side=Device gprbuild -m2 -P ../../../api/cuda_device.gpr -j0
	Execution_Side=Device gprbuild -m2 -P device.gpr -j0
	Execution_Side=Host gprbuild -m2 -P host -j0 -largs $(PWD)/obj/device/*.fatbin.o
