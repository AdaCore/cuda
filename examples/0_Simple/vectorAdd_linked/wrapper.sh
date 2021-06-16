#!/bin/sh -e

# PATH of the example to run. Should be e.g. examples/0_Simple/vectorAdd/
EXAMPLE_DIRECTORY="$PWD"
CUDA_BIN_DIR="/usr/local/cuda/bin/"
# Min compute capability. E.g. if you're on sm_75, SM_XX should be 75
SM_XX="75"

OBJ="$EXAMPLE_DIRECTORY/obj"
SRC="$EXAMPLE_DIRECTORY/src"

KERNEL_GLOBAL_SRC_NAME="kernel_global.adb"
KERNEL_GLOBAL_SRC_PATH="$SRC/$KERNEL_GLOBAL_SRC_NAME"
KERNEL_GLOBAL_PTX_NAME="${KERNEL_GLOBAL_SRC_NAME%.*}.s"
KERNEL_GLOBAL_PTX_PATH="$OBJ/$KERNEL_GLOBAL_PTX_NAME"
KERNEL_GLOBAL_CUBIN_NAME="${KERNEL_GLOBAL_SRC_NAME%.*}.cubin"
KERNEL_GLOBAL_CUBIN_PATH="$OBJ/$KERNEL_GLOBAL_CUBIN_NAME"

mkdir -p "$OBJ"

######################################
# BUILD FAT BINARY FOR KERNEL_GLOBAL #
######################################

"llvm-gcc" \
   -I"$EXAMPLE_DIRECTORY/../../../api/device_static/" \
   -O2 -S -gnatp -gnatn -mcpu=sm_"$SM_XX" --target=nvptx64 \
   "$KERNEL_GLOBAL_SRC_PATH" -o "$KERNEL_GLOBAL_PTX_PATH"

# Create CUBIN
"$CUDA_BIN_DIR/ptxas" -arch=sm_"$SM_XX" -m64 --compile-only "$KERNEL_GLOBAL_PTX_PATH" --output-file "$KERNEL_GLOBAL_CUBIN_PATH" 

#######################################
## BUILD FAT BINARY FOR KERNEL_DEVICE #
#######################################

KERNEL_DEVICE_SRC_NAME="kernel_device.adb"
KERNEL_DEVICE_SRC_PATH="$SRC/$KERNEL_DEVICE_SRC_NAME"
KERNEL_DEVICE_PTX_NAME="${KERNEL_DEVICE_SRC_NAME%.*}.s"
KERNEL_DEVICE_PTX_PATH="$OBJ/$KERNEL_DEVICE_PTX_NAME"
KERNEL_DEVICE_CUBIN_NAME="${KERNEL_DEVICE_SRC_NAME%.*}.cubin"
KERNEL_DEVICE_CUBIN_PATH="$OBJ/$KERNEL_DEVICE_CUBIN_NAME"

"llvm-gcc" \
   -I"$EXAMPLE_DIRECTORY/../../../api/device_static/" \
   -O2 -S -gnatp -gnatn -mcpu=sm_"$SM_XX" --target=nvptx64 \
   "$KERNEL_DEVICE_SRC_PATH" -o "$KERNEL_DEVICE_PTX_PATH"

# Create CUBIN
"$CUDA_BIN_DIR/ptxas" -arch=sm_"$SM_XX" -m64 --compile-only "$KERNEL_DEVICE_PTX_PATH" --output-file "$KERNEL_DEVICE_CUBIN_PATH" 

##############################
## LINK FATBINARIES TOGETHER #
##############################

LINKED_CUBIN_NAME="kernel_global.sm_$SM_XX.cubin"
LINKED_CUBIN_PATH="$OBJ/$LINKED_CUBIN_NAME"
LINKED_CUBIN_FATBIN_NAME="kernel_global.fatbin"
LINKED_CUBIN_FATBIN_PATH="$OBJ/$LINKED_CUBIN_FATBIN_NAME"
LINKED_CUBIN_FATBIN_OBJ_NAME="$LINKED_CUBIN_FATBIN_NAME.o"

# Link kernels together
"$CUDA_BIN_DIR/nvlink" --arch=sm_"$SM_XX" -m64  "-L$CUDA_BIN_DIR/../targets/x86_64-linux/lib/stubs" "-L$CUDA_BIN_DIR/../targets/x86_64-linux/lib" "$KERNEL_DEVICE_CUBIN_PATH" "$KERNEL_GLOBAL_CUBIN_PATH"  -lcudadevrt  -o "$LINKED_CUBIN_PATH"

# Create fatbinary out of linked cubin
fatbinary --create="$LINKED_CUBIN_FATBIN_PATH" -64 -link "--image3=kind=elf,sm=75,file=$LINKED_CUBIN_PATH"

# Create an object out of the fatbinary. Note: we need to move to the object
# directory to get ld to produce a predictable name.
CWD_BACKUP="$(pwd)"
cd "$(dirname "$LINKED_CUBIN_FATBIN_PATH")"
ld -r -b binary "$LINKED_CUBIN_FATBIN_NAME" -o "$LINKED_CUBIN_FATBIN_OBJ_NAME"
cd "$CWD_BACKUP"

gprbuild -Pmain -largs kernel_global.fatbin.o
