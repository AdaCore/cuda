# no set -e because this is supposed to be sourced!

# https://stackoverflow.com/a/28776166
# Cannot detect if the script is being sourced from within a script
# in that case set the env var NO_SOURCED_CHECK
is_sourced() {
    if [ -n "$ZSH_VERSION" ]; then
        case $ZSH_EVAL_CONTEXT in *:file:*) return 0;; esac
    else  # Add additional POSIX-compatible shell names here, if needed.
        case ${0##*/} in dash|-dash|bash|-bash|ksh|-ksh|sh|-sh) return 0;; esac
    fi
    return 1  # NOT sourced.
}

if [ -z "$NO_SOURCED_CHECK" ]; then
    if ! is_sourced || ! [ -f $PWD/env.sh ] ; then
        echo "This script is meant to be sourced from its own directory"
        exit 2
    fi
fi

CURRENT=$(pwd)
ROOT="$CURRENT/.."

# being sourced, must be super careful with error return value
CUDA_ROOT=$($SHELL $CURRENT/locate_cuda_root.sh) || return 2
export CUDA_ROOT # direct export would gobble up eventual error

export GPR_PROJECT_PATH="$ROOT/cuda/api/install/share/gpr:$ROOT/uwrap/lang_template/build:$ROOT/uwrap/lang_test/build:$GPR_PROJECT_PATH:$ROOT/gnat-llvm/share/gpr"
export PYTHONPATH="$ROOT/uwrap/lang_template/build/python:$ROOT/uwrap/lang_test/build/python:$PYTHONPATH"
export LD_LIBRARY_PATH="$ROOT/uwrap/lang_template/build/lib/relocatable/dev:$ROOT/uwrap/lang_test/build/lib/relocatable/dev:$LD_LIBRARY_PATH:$ROOT/gnat-llvm/lib"
export PATH="$ROOT/llvm-ads/bin:$ROOT/uwrap/bin:$ROOT/uwrap/lang_template/build/lib/relocatable/dev:$ROOT/uwrap/lang_template/build/obj-mains:$ROOT/uwrap/lang_template/build/scripts:$ROOT/uwrap/lang_test/build/lib/relocatable/dev:$ROOT/uwrap/lang_test/build/obj-mains:$ROOT/uwrap/lang_test/build/scripts:$PATH:$ROOT/gnat-llvm/bin"
export C_INCLUDE_PATH="$ROOT/uwrap/lang_template/build:$ROOT/uwrap/lang_test/build:$C_INCLUDE_PATH"
export DYLD_LIBRARY_PATH="$ROOT/uwrap/lang_template/build/lib/relocatable/dev:$ROOT/uwrap/lang_test/build/lib/relocatable/dev:$DYLD_LIBRARY_PATH"
export MYPYPATH="$ROOT/uwrap/lang_template/build/python:$ROOT/uwrap/lang_test/build/python:$MYPYPATH"
