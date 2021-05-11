export CUDA_PATH=$CUDA_ROOT/include/

rm -rf host device

mkdir host
cd host
rm -rf cuda_api cuda_raw_binding
mkdir cuda_api cuda_raw_binding
cd cuda_raw_binding
gcc -c -fdump-ada-spec "$CUDA_PATH/cuda_runtime_api.h"
echo "project CUDA_Raw is end CUDA_Raw;" > cuda_raw.gpr
cd ../cuda_api
uwrap -l ada -w ../../cuda.wrp ../cuda_raw_binding/*.ads -P../cuda_raw_binding/cuda_raw
gnatpp *
cd ../..

mkdir device
cd device
rm -rf cuda_api cuda_raw_binding
mkdir cuda_api cuda_raw_binding
cd cuda_raw_binding
#g++ -c -fdump-ada-spec -D __CUDACC__ -D __CUDA_ARCH__ "$CUDA_PATH/device_functions.h"
g++ -c -fdump-ada-spec -D __CUDACC__ -D __CUDA_ARCH__ "$CUDA_PATH/cuda_runtime_api.h"
echo "project CUDA_Raw is end CUDA_Raw;" > cuda_raw.gpr
cd ../cuda_api
uwrap -l ada -w ../../cuda.wrp ../cuda_raw_binding/*_h.ads -P../cuda_raw_binding/cuda_raw
gnatpp *
cd ../..
