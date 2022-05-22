build:
	gprbuild -P device.gpr -v
	gprbuild -P host -j0 -largs $(PWD)/lib/*.fatbin.o
