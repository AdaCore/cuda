#!/bin/sh
set -ex

CURRENT_FILE=$(realpath "$0")
CURRENT_DIR=$(dirname "$CURRENT_FILE")
ROOT=$(dirname "$CURRENT_DIR")

# shellcheck disable=SC1090 # constant: parent dir
# shellcheck disable=SC1091 # FIXME shellcheck -P SCRIPTDIR
# https://github.com/koalaman/shellcheck/issues/769#issuecomment-486492469
. "$ROOT/locate_cuda_root.sh"

export CUDA_PATH="$CUDA_ROOT/include/"
test -d "$CUDA_PATH"

assert_has_files() {
    if [ -z "$(ls "$1" 2>/dev/null)" ]; then
        echo "uwrap failed: no file in $1" >&2
        return 1
    fi
}

assert_in_path() {
    if ! command -v "$1">/dev/null; then
        echo "Exec not found: $1" >&2
        return 1
    fi
}

case $CUDA_HOST in
    aarch64-linux) compiler_driver=aarch64-linux-gnu-gcc; uwrap_target_switch="--target=aarch64-linux-gnu"; gnatls_name="aarch64-linux-gnu-gnatls";;
    *) compiler_driver=gcc; uwrap_target_switch=""; gnatls_name="gnatls";;
esac

# Pre for uwrap invocations
assert_in_path $gnatls_name

rm -rf host device

echo "Generating host binding for $CUDA_PATH/cuda_runtime_api.h"
mkdir host
cd host
rm -rf cuda_api cuda_raw_binding
mkdir cuda_api cuda_raw_binding
cd cuda_raw_binding
$compiler_driver -c -fdump-ada-spec "$CUDA_PATH/cuda_runtime_api.h" -w
echo "project CUDA_Raw is end CUDA_Raw;" > cuda_raw.gpr
cd ../cuda_api
uwrap "$uwrap_target_switch" -l ada -w ../../cuda-host.wrp ../cuda_raw_binding/*.ads -P../cuda_raw_binding/cuda_raw
assert_has_files "$PWD"
gnatpp ./*
cd ../..

# This is a temporary workaround for a binding generation error. The function on which
# binding generation fails is cudaGraphAddNode_v2, which was introduced with CUDA 12.3.
# It requires perl, so systems that have CUDA 12.3 or later but do not have perl
# require manual interventions.
if command -v perl /dev/null; then
    perl -0777 -i -p hotfix.pl host/cuda_api/cuda-runtime_api.adb
fi

echo "Generating device binding for $CUDA_PATH/cuda_runtime_api.h"
mkdir device
cd device
rm -rf cuda_api cuda_raw_binding
mkdir cuda_api cuda_raw_binding
cd cuda_raw_binding
$compiler_driver -c -fdump-ada-spec -D __CUDACC__ -D __CUDA_ARCH__ "$CUDA_PATH/cuda_runtime_api.h" -w
echo "project CUDA_Raw is end CUDA_Raw;" > cuda_raw.gpr
cd ..
mkdir libdevice
cd libdevice
LIBDEVICE_PATH="$(find -L "$CUDA_ROOT" -iname "libdevice.*.bc" | head -n 1)"
llvm-ads "$LIBDEVICE_PATH" "$PWD"/libdevice.ads
gnatpp ./*
cd ../..
