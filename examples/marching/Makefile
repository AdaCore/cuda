build:
	gprbuild -P device.gpr -j0
	gprbuild -P host.gpr -j0 -largs $(PWD)/lib/*.fatbin.o
