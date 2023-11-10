#!/bin/bash

# inside docker container
export TRTLLM_HOME="/code/tensorrt_llm"
export TOKENIZER="/models/Codellama-34b-instruct-hf/"
export MODELNAME="CodeLlama-34b-instruct-hf"
export ENGINES="$TRTLLM_HOME/engines"

export MAX_INPUT=2560
export MAX_OUTPUT=2560
export MAX_BATCH=1
