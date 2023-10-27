# shellcheck shell=sh
# this file is SUPPOSED TO BE SOURCED
# so, no shebang, no set -e

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
    if ! is_sourced || ! [ -f "$PWD/env.sh" ] ; then
        echo "This script is meant to be sourced from its own directory"
        exit 2
    fi
fi

CURRENT=$(pwd)
ROOT="$CURRENT/.."

# being sourced, must be super careful with error return value
CUDA_ROOT=$("$SHELL" "$CURRENT/locate_cuda_root.sh") || return 2
export CUDA_ROOT # direct export would gobble up eventual error

export GPR_PROJECT_PATH="$ROOT/cuda/api/install/share/gpr"
export PATH="$ROOT/llvm-ads/bin:$ROOT/uwrap/bin:$PATH:$ROOT/gnat-llvm/bin"
