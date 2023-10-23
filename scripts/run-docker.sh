export SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
export TRTLLM_HOME=$SCRIPT_DIR/..

docker run --gpus all --ipc=host --ulimit memlock=-1 --shm-size=20g --rm -it -v $TRTLLM_HOME:/code/tensorrt_llm -v /opt/models:/models -w /code/tensorrt_llm tensorrt_llm/release bash
