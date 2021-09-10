export PATH := install/bin:$(PATH)

BB_SRC   := ../bb-runtimes
GNAT_SRC := ../gnat

local_llvm := $(shell which llvm-gcc)
llvm_dir   := $(shell dirname $(dir $(local_llvm)))

.PHONY: main install clean


main: install/bin
	@echo $(PATH)
	gprbuild -p -P wrapper/wrapper.gpr
	cp wrapper/obj/gnatcuda_wrapper install/bin/cuda-gcc
	cp install/bin/cuda-gcc $(llvm_dir)/bin/cuda-gcc
	./gen-rts-sources.py --bb-dir $(BB_SRC) --gnat $(GNAT_SRC) --rts-profile=light
	./build-rts.py --bb-dir $(BB_SRC) --rts-src-descriptor install/lib/gnat/rts-sources.json cuda-device --force -b
	mv install/device-cuda install/lib/rts-device-cuda
	cp -R install/lib/rts-device-cuda $(llvm_dir)/lib/rts-device-cuda

install/bin:
	mkdir install
	mkdir install/bin

uninstall:
	rm $(llvm_dir)/bin/cuda-gcc
	rm -rf $(llvm_dir)/lib/rts-device-cuda

clean:
	rm -rf install
	gprclean -P wrapper/wrapper.gpr
