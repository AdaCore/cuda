build:
	gprbuild -Xcubin_version=sm_75 -Xhost_target=x86_64-linux -P device.gpr -v
	gprbuild -Xhost_target=x86_64-linux -P host -j0 -largs $(PWD)/lib/*.fatbin.o
