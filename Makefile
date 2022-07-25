export PATH := install/bin:$(PATH)

BB_SRC   := ../bb-runtimes
GNAT_SRC := ../gnat

local_llvm := $(shell which llvm-gcc)
llvm_dir   := $(shell dirname $(dir $(local_llvm)))

.PHONY: main clean wrapper runtime


main: install/bin wrapper runtime

wrapper:
	@echo "======================= WRAPPER BUILDING"
	@echo $(PATH)
	gprbuild -p -P wrapper/wrapper.gpr
	cp wrapper/obj/gnatcuda_wrapper install/bin/cuda-gcc
	cp install/bin/cuda-gcc $(llvm_dir)/bin/cuda-gcc

runtime: libdevice.ads
	@echo "======================= RUNTIME BUILDING"
	rm -rf install/include/rts-sources/device_gnat
	./gen-rts-sources.py --bb-dir $(BB_SRC) --gnat $(GNAT_SRC) --rts-profile=light
	./build-rts.py --bb-dir $(BB_SRC) --rts-src-descriptor install/lib/gnat/rts-sources.json cuda-device  --force -b
	rm -rf install/lib/rts-device-cuda
	mv install/device-cuda install/lib/rts-device-cuda
	cp -R runtime/device_gnat/* install/lib/rts-device-cuda/gnat/
	rm -rf $(llvm_dir)/lib/rts-device-cuda
	cp -R install/lib/rts-device-cuda $(llvm_dir)/lib/rts-device-cuda


libdevice.ads:
	llvm-ads $(shell find -L /usr/local/cuda -iname "libdevice.*.bc" | head -n 1) ./runtime/device_gnat/libdevice.ads

install/bin:
	@echo "======================= INSTALL SETUP"
	mkdir -p install
	mkdir -p install/bin

uninstall:
	rm $(llvm_dir)/bin/cuda-gcc
	rm -rf $(llvm_dir)/lib/rts-device-cuda

clean:
	rm -rf install
	gprclean -P wrapper/wrapper.gpr
