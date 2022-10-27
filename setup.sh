#! /bin/sh
set -e


while [ $# -gt 0 ] ; do
  case $1 in
    -mcpu) GPU_ARCH="$2" ;;
  esac
  shift
done

if [ -z $GPU_ARCH ]; then
    echo "Target CPU not specified"
    echo "Syntax:"
    echo "$> sh setup.sh -mcpu <gpu architecture>"
    echo "For example:"
    echo "$> sh setup.sh -mcpu sm_75"
    return 1
fi

ROOT=$(dirname $(readlink -f "$0"))

(
echo ""
echo "Starting setup GNAT for CUDA"
echo "============================"
echo ""
cd $ROOT
NO_SOURCED_CHECK=1 . ./env.sh
echo "CUDA installation detected on $CUDA_ROOT"
echo ""
echo "Generating Ada runtime for your CUDA installation"
echo "================================================="
echo ""
make runtime GPU_ARCH=$GPU_ARCH
echo "GPU_ARCH=$GPU_ARCH" >> Makefile.env
echo ""
echo "Generating Ada bindings for your CUDA installation"
echo "=================================================="
echo ""
(
    cd api
    sh bind.sh
)
echo ""
echo "Post setup notes"
echo "================"
echo "Please source env.sh to get environment setup"
echo "Please re-run this script after CUDA installation updates"
echo ""
)
