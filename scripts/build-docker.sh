export SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
export TRTLLM_HOME=$SCRIPT_DIR/..

cd $TRTLLM_HOME
make -C docker release_build CUDA_ARCHS="90-real"
