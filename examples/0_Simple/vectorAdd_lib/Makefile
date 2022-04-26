build:
	rm -f lib/*
	rm -f obj/device/*
	#Execution_Side=Device gprbuild -P ../../../api/cuda_device.gpr
	Execution_Side=Device gprbuild -P device.gpr -v
	Execution_Side=Host gprbuild -P host -j0 -largs $(PWD)/lib/*.fatbin.o
