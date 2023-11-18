#!/bin/bash

TRTLLM_HOME=/path/to/repo  # the TRT-LLM github repo
TRT_RESULTS=/path/to/dir  # the directory where you'd like results saved
MODELS_DIR=/path/to/models  # the parent directory where models are stored


IMAGE="tensorrt_llm/release"
#IMAGE="tensorrt_llm/installed"

docker run --gpus all --ipc=host --ulimit memlock=-1 --shm-size=20g --rm -it -v $TRTLLM_HOME:/code/tensorrt_llm -v $MODELS_DIR:/models -v /opt/bin:/opt/bin -v /tmp/cuda-reservations:/tmp/cuda-reservations -v $TRT_RESULTS:/code/results -w /code/tensorrt_llm/scripts/p5d $IMAGE bash

