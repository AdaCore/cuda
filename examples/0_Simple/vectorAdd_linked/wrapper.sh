#!/bin/sh -e

set -e

CUDA_BIN_DIR="/usr/local/cuda/bin/"
SM_XX="75"

# Build CUDA runtime
Execution_Side=Device gprbuild -P ../../../api/cuda_device.gpr -j0

mkdir -p obj/device obj/host

# Build fatbin
"llvm-gcc" \
   -I"/../../../api/device_static/" \
   -O2 -S -gnatp -gnatn -mcpu=sm_"$SM_XX" --target=nvptx64 \
   src/common/kernel.adb -o obj/device/kernel.s

ptxas -arch=sm_"$SM_XX" -m64 --compile-only \
   obj/device/kernel.s --output-file obj/device/kernel.cubin

"llvm-gcc" \
   -I"../../../api/device_static/" \
   -O2 -S -gnatp -gnatn -mcpu=sm_"$SM_XX" --target=nvptx64 \
   src/common/device_functions.adb -o obj/device/device_functions.s

ptxas -arch=sm_"$SM_XX" -m64 --compile-only \
   obj/device/device_functions.s --output-file obj/device/device_functions.cubin

nvlink --arch=sm_"$SM_XX" -m64  \
   -L/usr/local/cuda/targets/x86_64-linux/lib/stubs \
   -L/usr/local/cuda/targets/x86_64-linux/lib \
   obj/device/device_functions.cubin obj/device/kernel.cubin \
   -lcudadevrt -o obj/device/linked_cubin

fatbinary --create=obj/device/kernel.fatbin -64 -link --image3=kind=elf,sm=75,file=obj/device/linked_cubin

(
cd obj/device
ld -r -b binary kernel.fatbin -o kernel.fatbin.o
)

# Build host
Execution_Side=Host gprbuild -P host -j0 -largs $(pwd)/obj/device/*.fatbin.o
