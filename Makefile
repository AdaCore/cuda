main:
	rm -rf install
	mkdir install
	gprbuild -P wrapper/wrapper.gpr
	mkdir install/bin
	cp wrapper/obj/gnatcuda_wrapper install/bin/cuda-gcc	
	mkdir install/lib
	mkdir install/lib/gnat
	mkdir install/lib/gnat/cuda-full
	cp -r runtime/adainclude install/lib/gnat/cuda-full
	mkdir install/lib/gnat/cuda-full/adalib