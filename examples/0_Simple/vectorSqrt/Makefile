build:
	Execution_Side=Device gprbuild --keep-temp-files -P ../../../api/cuda_device.gpr -j0
	Execution_Side=Device gprbuild --keep-temp-files -P device.gpr -j0
	Execution_Side=Host gprbuild -P host -j0 -largs $(PWD)/obj/device/*.fatbin.o
