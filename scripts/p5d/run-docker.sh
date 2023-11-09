#!/bin/bash

if [[ -n "$TRTLLM_HOME" ]] ; then
    echo "TRTLLM_HOME unset"
    exit 1
fi

docker run --gpus all --ipc=host --ulimit memlock=-1 --shm-size=20g --rm -it -v $TRTLLM_HOME:/code/tensorrt_llm -v /opt/dlami/nvme/models/:/models/ -w /code/tensorrt_llm tensorrt_llm/release bash

