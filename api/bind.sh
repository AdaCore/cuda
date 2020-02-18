export CUDA_PATH="/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include"

gprbuild ../GNATwrap/gnatwrap.gpr
rm -rf cuda_api cuda_raw_binding
mkdir cuda_api cuda_raw_binding
cd cuda_raw_binding
gcc -c -fdump-ada-spec "$CUDA_PATH/cuda_runtime_api.h"
echo "project CUDA_Raw is end CUDA_Raw;" > cuda_raw.gpr
cd ../cuda_api
../../GNATwrap/obj/gnatwrap-driver ../cuda_raw_binding/*.ads -P../cuda_raw_binding/cuda_raw
gnatpp  --insert-blank-lines --max-line-length=300 *