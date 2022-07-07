CURRENT=$(dirname $0)
ROOT="$CURRENT/.."

. $CURRENT/locate_cuda_root.sh >/dev/null

export GPR_PROJECT_PATH="$ROOT/cuda/api/:$ROOT/uwrap/lang_template/build:$ROOT/uwrap/lang_test/build:$ROOT/gnat-llvm/share/gpr:$GPR_PROJECT_PATH"
export PYTHONPATH="$ROOT/uwrap/lang_template/build/python:$ROOT/uwrap/lang_test/build/python:$PYTHONPATH"
export LD_LIBRARY_PATH="$ROOT/uwrap/lang_template/build/lib/relocatable/dev:$ROOT/uwrap/lang_test/build/lib/relocatable/dev:$ROOT/gnat-llvm/lib:$LD_LIBRARY_PATH"
export PATH="$ROOT/uwrap/bin:$ROOT/uwrap/lang_template/build/lib/relocatable/dev:$ROOT/uwrap/lang_template/build/obj-mains:$ROOT/uwrap/lang_template/build/scripts:$ROOT/uwrap/lang_test/build/lib/relocatable/dev:$ROOT/uwrap/lang_test/build/obj-mains:$ROOT/uwrap/lang_test/build/scripts:$ROOT/gnat-llvm/bin:$PATH"
export C_INCLUDE_PATH="$ROOT/uwrap/lang_template/build:$ROOT/uwrap/lang_test/build:$C_INCLUDE_PATH"
export DYLD_LIBRARY_PATH="$ROOT/uwrap/lang_template/build/lib/relocatable/dev:$ROOT/uwrap/lang_test/build/lib/relocatable/dev:$DYLD_LIBRARY_PATH"
export MYPYPATH="$ROOT/uwrap/lang_template/build/python:$ROOT/uwrap/lang_test/build/python:$MYPYPATH"
