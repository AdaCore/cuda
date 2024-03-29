export PATH := install/bin:$(PATH)

# Path to BB runtime's source repo
BB_SRC   := $(realpath ../bb-runtimes)
ifeq (, $(BB_SRC))
 $(error "Could not locate BB-Runtimes' directory")
endif
# Path to GNAT's source repo
GNAT_SRC := $(realpath ../gnat)
ifeq (, $(GNAT_SRC))
 $(error "Could not locate GNAT source's directory")
endif

local_llvm := $(shell which llvm-gcc)
ifeq (, $(local_llvm))
 $(error "No llvm-gcc in PATH")
endif
$(info "LLVM's GCC    : $(local_llvm)")

llvm_dir   := $(shell dirname $(dir $(local_llvm)))
ifeq (, $(llvm_dir))
 $(error "Could not locate LLVM's directory")
endif
$(info "LLVM directory: $(llvm_dir)")

cuda_dir := $(shell sh locate_cuda_root.sh)
ifeq (, $(cuda_dir))
 $(error "Could not locate CUDA's directory")
endif
$(info "CUDA directory: $(cuda_dir)")

libdevice.bc := $(shell find -L $(cuda_dir) -iname "libdevice.*.bc" | head -n 1)
ifeq (, $(libdevice.bc))
 $(error "Could not locate libdevice.*.bc")
endif
$(info "libdevice.bc  : $(libdevice.bc)")

export PATH := $(cuda_dir)/bin:$(PATH)

.PHONY: main clean runtime

main: install/bin runtime

runtime:
	@echo "======================= RUNTIME BUILDING"
	rm -rf install/include/rts-sources/device_gnat
	./gen-rts-sources.py --bb-dir $(BB_SRC) --gnat $(GNAT_SRC)/src/ada --rts-profile=light
	./build-rts.py --bb-dir $(BB_SRC) --rts-src-descriptor install/lib/gnat/rts-sources.json cuda-device  --force -b --mcpu $(GPU_ARCH)
	rm -rf install/lib/rts-device-cuda
	mv install/device-cuda install/lib/rts-device-cuda
	cp -R runtime/device_gnat/* install/lib/rts-device-cuda/gnat/
	rm -rf $(llvm_dir)/lib/rts-device-cuda
	cp -p install/include/rts-sources/device_gnat/* install/lib/rts-device-cuda/gnat/
	cp -pR install/lib/rts-device-cuda $(llvm_dir)/lib/rts-device-cuda

install/bin:
	@echo "======================= INSTALL SETUP"
	mkdir -p install
	mkdir -p install/bin

uninstall:
	rm $(llvm_dir)/bin/cuda-gcc
	rm -rf $(llvm_dir)/lib/rts-device-cuda

clean:
	rm -rf install
