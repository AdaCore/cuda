export CUDA_PATH=$CUDA_ROOT/include/

rm -rf cuda_api cuda_raw_binding
mkdir cuda_api cuda_raw_binding
cd cuda_raw_binding
gcc -c -fdump-ada-spec "$CUDA_PATH/cuda_runtime_api.h"
echo "project CUDA_Raw is end CUDA_Raw;" > cuda_raw.gpr
cd ../cuda_api
UWRAP_PREFIX=../../../uwrap/
$UWRAP_PREFIX/obj/uwrap -l ada -w ../cuda.wrp ../cuda_raw_binding/*.ads -P../cuda_raw_binding/cuda_raw
gnatpp --max-line-length=80 *
