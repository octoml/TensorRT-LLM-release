#!/bin/bash

TRTLLM_HOME="/home/joseph/workspace/TensorRT-LLM-release"
RESULTS="/home/joseph/results"
#IMAGE="tensorrt_llm/release"
IMAGE="tensorrt_llm/installed"

docker run --gpus all --ipc=host --ulimit memlock=-1 --shm-size=20g --rm -it -v $TRTLLM_HOME:/code/tensorrt_llm -v /opt/dlami/nvme/models:/models -v /opt/bin:/opt/bin -v /tmp/cuda-reservations:/tmp/cuda-reservations -v $RESULTS:/code/results -w /code/tensorrt_llm/scripts/p5d $IMAGE bash

