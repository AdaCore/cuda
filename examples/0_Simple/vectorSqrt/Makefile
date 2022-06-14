build:
	Execution_Side=Device gprbuild -m2 -P ../../../api/cuda_device.gpr -j0
	Execution_Side=Device gprbuild -Xcubin_version=sm_75 -Xhost_target=x86_64-linux -m2 -P device
	Execution_Side=Host gprbuild -Xhost_target=x86_64-linux -m2 -P host -largs $(PWD)/lib/*.fatbin.o