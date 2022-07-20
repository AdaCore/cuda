#! /bin/sh
set -e

usage() {
    echo "usage: $(basename $0) [-h] [cuda_root_directory]"
    echo "Tests and returns the path to the CUDA root, from either the "
    echo "environment, the argument given, or a very smart (not) heuristic"
    echo ""
    echo "optional arguments"
    echo "  -h                     Display usage"
    echo "  cuda_root_directory    Directory to the CUDA install"
    echo ""
    echo "Output and exit status:"
    echo "  In case the directory is invalid, return an error code."
    echo "  Otherwise outputs the name of the directory, and exports"
    echo "  the CUDA_ROOT environment variable to the directory."
}

assert() {
    set +e
    # Sub-shell for pipes...
    $SHELL -ec "$*" >/dev/null 2>&1
    fail=$?
    set -e
    if [ $fail -eq 1 ]; then
        echo "assert failed: $@" >&2
        exit 2
    fi
}

# Check args and display usage
if [ $# -gt 1 ]; then
    echo "wrong number of arguments">&2
    echo
    usage
    exit 2
fi

if [ "$1" = "-h" ]; then
    usage
    exit 0
fi

# Locate root
if [ ! -z "$1" ]; then
    ## Use argument
    CUDA_ROOT="$1"
elif [ ! -z "$CUDA_ROOT" ]; then
    ## Use already set value
    break
elif [ ! -z "$(which nvcc)" ]; then
    ## Heuristic: $CUDA_ROOT/bin/nvcc
    nvcc = $(readlink -f $(which nvcc))
    assert test -f "$nvcc"
    CUDA_ROOT="$(dirname $(dirname $nvcc))"
else
    ## Try a "standard" directory
    CUDA_ROOT="/usr/local/cuda"
fi

# Check root seems correct
assert test -d $(realpath "$CUDA_ROOT")
assert test -d "$CUDA_ROOT/include"
assert "find -L '$CUDA_ROOT' -iname 'libdevice.*.bc' -print -quit | grep lib" 

# Result
export CUDA_ROOT
echo "$CUDA_ROOT"
