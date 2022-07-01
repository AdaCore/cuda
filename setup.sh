echo ""
echo "Starting setup GNAT for CUDA"
echo "============================"
echo ""
. ./env.sh
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