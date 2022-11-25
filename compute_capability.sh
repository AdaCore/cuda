set -e

usage() {
    echo "usage: $(basename $0) [--expect-single | -s] [--no-expect-single | -S] [--compute-prefix | -c] [--sm-prefix | -C]"
}

expect_single=0
compute_prefix=1
while [ ! -z $1 ]; do
    case $1 in
    -c|--compute-prefix)
        compute_prefix=1
        ;;
    -C|--sm-prefix)
        compute_prefix=0
        ;;
    -s|--expect-single)
        expect_single=1
        ;;
    -S|--no-expect-single)
        expect_single=0
        ;;
    -h|--help)
        usage
        exit 0
        ;;
    *)
        echo "unknown arg $1" >&2
        usage
        exit 2
        ;;
    esac

    shift
done

cap=$(nvidia-smi --query-gpu=compute_cap --format=csv | tail -n +2 | tr -d .)
number=$(echo "$cap" | wc -l)

# number of GPU
if [ $expect_single -eq 0 ]; then
    echo "$number GPU"
elif [ $number -ne 1 ]; then
    echo "expected a single GPU, found $number" >&2
    exit 1
fi

# capacity
echo "$cap" | while read c; do
    if [ $compute_prefix -eq 0 ]; then
        prefix="sm_"
    else
        prefix="compute_"
    fi
    echo "$prefix$c"
done
