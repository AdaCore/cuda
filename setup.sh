#! /bin/bash
set -e

fatal() {
    echo "$*" >&2
    exit 1
}

if [ ! -f ./env.sh ]; then
    fatal "$(basename "$0") must be run from the CUDA directory"
fi

if ! command -v gprbuild 1>/dev/null; then
    fatal "gprbuild is required"
fi

gprbuild_version=$(gprbuild --version | head -n 1 | cut -d ' ' -f 3)
gprbuild_major_version=$(echo "${gprbuild_version}" | cut -d '.' -f 1)

if [ "$gprbuild_major_version" -lt 23 ]; then
    fatal "gprbuild 23.0 or later is required, only ${gprbuild_version} was found"
fi

while [ $# -gt 0 ] ; do
  case $1 in
    -mcpu) GPU_ARCH="$2" ;;
  esac
  shift
done


if [ -z "$GPU_ARCH" ]; then
    # shellcheck disable=SC2039 # no POSIX `-n` switch, at worst it's echoed
    echo -n "autodetect compute capability: "
    GPU_ARCH=$(\
        sh ./compute_capability.sh --expect-single --sm-prefix \
        || true \
    )
    if [ -z "$GPU_ARCH" ]; then
        echo "FAIL"

        fatal "$(cat <<EOF
Target GPU detection failed, and no GPU was specified\n
Please manually specify a target GPU with -mcpu\n
\n
Syntax:\n
$> sh setup.sh -mcpu <gpu architecture>\n
For example:\n
$> sh setup.sh -mcpu sm_75
EOF
        )"
    fi

    echo "OK: $GPU_ARCH"
fi

ROOT=$(dirname "$(readlink -f "$0")")

(
echo ""
echo "Starting setup GNAT for CUDA"
echo "============================"
echo ""

cd "$ROOT"
# shellcheck disable=SC1091 # FIXME shellcheck -P SCRIPTDIR
# https://github.com/koalaman/shellcheck/issues/769#issuecomment-486492469
NO_SOURCED_CHECK=1 . ./env.sh

echo "CUDA installation detected on $CUDA_ROOT"
echo ""

echo "Generating Ada runtime for your CUDA installation"
echo "================================================="
echo ""
make runtime GPU_ARCH="$GPU_ARCH"
echo ""

echo "GPU_ARCH=$GPU_ARCH" > Makefile.env

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

echo "Installing the CUDA API"
echo "======================="
echo ""
(
    # shellcheck disable=SC1091 # FIXME shellcheck -P SCRIPTDIR
    # https://github.com/koalaman/shellcheck/issues/769#issuecomment-486492469
    NO_SOURCED_CHECK=1 . ./env.sh
    cd api
    rm -rf install
    for gpr in cuda_api_device.gpr cuda_api_host.gpr; do
        gprbuild -P $gpr
        gprinstall -P $gpr \
            --prefix=install \
            --create-missing-dirs
    done
    cp architecture.gpr install/share/gpr
)
echo ""

echo "Post setup notes"
echo "================"
echo ""
echo "Please source env.sh to get environment setup"
echo "Please re-run this script after CUDA installation updates"
echo ""
)
