set -ex

CURRENT_DIR=$(dirname $(realpath $0))
ROOT=$(dirname $CURRENT_DIR)

. "$ROOT/locate_cuda_root.sh"

export CUDA_PATH=$CUDA_ROOT/include/
test -d "$CUDA_PATH"

assert_has_files() {
    if [ -z $(ls $1 2>/dev/null) ]; then
        echo "uwrap failed: no file in $1" >&2
        return 1
    fi
}

rm -rf host device

echo "Generating host binding for $CUDA_PATH/cuda_runtime_api.h"
mkdir host
cd host
rm -rf cuda_api cuda_raw_binding
mkdir cuda_api cuda_raw_binding
cd cuda_raw_binding
gcc -c -fdump-ada-spec "$CUDA_PATH/cuda_runtime_api.h" -w
echo "project CUDA_Raw is end CUDA_Raw;" > cuda_raw.gpr
cd ../cuda_api
uwrap -l ada -w ../../cuda-host.wrp ../cuda_raw_binding/*.ads -P../cuda_raw_binding/cuda_raw
assert_has_files $PWD
gnatpp *
cd ../..

echo "Generating device binding for $CUDA_PATH/cuda_runtime_api.h"
mkdir device
cd device
rm -rf cuda_api cuda_raw_binding
mkdir cuda_api cuda_raw_binding
cd cuda_raw_binding
#g++ -c -fdump-ada-spec -D __CUDACC__ -D __CUDA_ARCH__ "$CUDA_PATH/device_functions.h"
g++ -c -fdump-ada-spec -D __CUDACC__ -D __CUDA_ARCH__ "$CUDA_PATH/cuda_runtime_api.h" -w
echo "project CUDA_Raw is end CUDA_Raw;" > cuda_raw.gpr
cd ..
mkdir libdevice
cd libdevice
LIBDEVICE_PATH="$(find -L "$CUDA_ROOT" -iname "libdevice.*.bc" | head -n 1)"
llvm-ads "$LIBDEVICE_PATH" "$PWD"/libdevice.ads
gnatpp *
cd ../..
