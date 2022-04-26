build:
	Execution_Side=Device gprbuild -P device.gpr -j0
	Execution_Side=Host gprbuild -P host.gpr -j0 -largs $(PWD)/lib/*.fatbin.o
