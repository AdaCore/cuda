#! /bin/sh
set -e

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
make runtime
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
