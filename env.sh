NVCC=`which nvcc`
export CUDA_ROOT=${NVCC%/*/*}

CUDA_ENV="$PWD/.."
ROOT="$PWD/.."

export GPR_PROJECT_PATH="$ROOT/api/:$CUDA_ENV/uwrap/lang_template/build:$CUDA_ENV/uwrap/lang_test/build:$CUDA_ENV/gnat-llvm/share/gpr:$GPR_PROJECT_PATH"
export PYTHONPATH="$CUDA_ENV/uwrap/lang_template/build/python:$CUDA_ENV/uwrap/lang_test/build/python:$PYTHONPATH"
export LD_LIBRARY_PATH="$CUDA_ENV/uwrap/lang_template/build/lib/relocatable/dev:$CUDA_ENV/uwrap/lang_test/build/lib/relocatable/dev:$CUDA_ENV/gnat-llvm/lib:$LD_LIBRARY_PATH"
export PATH="$CUDA_ENV/llvm-ads/bin:$CUDA_ENV/uwrap/bin:$CUDA_ENV/uwrap/lang_template/build/lib/relocatable/dev:$CUDA_ENV/uwrap/lang_template/build/obj-mains:$CUDA_ENV/uwrap/lang_template/build/scripts:$CUDA_ENV/uwrap/lang_test/build/lib/relocatable/dev:$CUDA_ENV/uwrap/lang_test/build/obj-mains:$CUDA_ENV/uwrap/lang_test/build/scripts:$CUDA_ENV/gnat-llvm/bin:$PATH"
export C_INCLUDE_PATH="$CUDA_ENV/uwrap/lang_template/build:$CUDA_ENV/uwrap/lang_test/build:$C_INCLUDE_PATH"
export DYLD_LIBRARY_PATH="$CUDA_ENV/uwrap/lang_template/build/lib/relocatable/dev:$CUDA_ENV/uwrap/lang_test/build/lib/relocatable/dev:$DYLD_LIBRARY_PATH"
export MYPYPATH="$CUDA_ENV/uwrap/lang_template/build/python:$CUDA_ENV/uwrap/lang_test/build/python:$MYPYPATH"
