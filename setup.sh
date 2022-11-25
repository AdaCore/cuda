#! /bin/sh
set -e

fatal() {
    echo $@ >&2
    return 1
}

if [ ! -f ./env.sh ]; then
    fatal "$(basename $0) must be run from the CUDA directory"
fi

while [ $# -gt 0 ] ; do
  case $1 in
    -mcpu) GPU_ARCH="$2" ;;
  esac
  shift
done

if [ -z $GPU_ARCH ]; then
    echo -n "autodetect compute capability: "
    GPU_ARCH=$(\
        sh ./compute_capability.sh --expect-single --sm-prefix \
        || true \
    )
    if [ -z "$GPU_ARCH" ]; then
        echo "FAIL"

        fatal $(cat <<EOF
Target GPU not specified\n
Syntax:\n
$> sh setup.sh -mcpu <gpu architecture>\n
For example:\n
$> sh setup.sh -mcpu sm_75
EOF
        )
        return 1
    fi

    echo "OK: $GPU_ARCH"
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
    echo "
project Architecture is

  for Languages use ();

  type GPU_Arch_Option is
    (\"sm_20\", --  Fermi
     \"sm_30\", --  Kepler
     \"sm_35\", --  Kepler
     \"sm_37\", --  Kepler
     \"sm_50\", --  Maxwell
     \"sm_52\", --  Maxwell
     \"sm_53\", --  Maxwell
     \"sm_60\", --  Pascal
     \"sm_61\", --  Pascal
     \"sm_62\", --  Pascal
     \"sm_70\", --  Volta
     \"sm_72\", --  Volta
     \"sm_75\", --  Turing
     \"sm_80\", --  Ampere
     \"sm_86\", --  Ampere
     \"sm_87\", --  Ampere
     \"sm_89\", --  Lovelace
     \"sm_90\"  --  Hopper
    );

    GPU_Arch : GPU_Arch_Option := \"$GPU_ARCH\";

end Architecture;" > architecture.gpr
    sh bind.sh
)
echo ""
echo "Post setup notes"
echo "================"
echo "Please source env.sh to get environment setup"
echo "Please re-run this script after CUDA installation updates"
echo ""
)
