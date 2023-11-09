#!/bin/bash

# inside docker container
export TRTLLM_HOME="/code/tensorrt_llm"
export TOKENIZER="/models/codellama/CodeLlama-34b-Instruct-hf/"
export MODELNAME="CodeLlama-34b-instruct-hf"
export ENGINES="$TRTLLM_HOME/engines"

export MAX_INPUT=2560
export MAX_OUTPUT=2560
export MAX_MATCH=1
