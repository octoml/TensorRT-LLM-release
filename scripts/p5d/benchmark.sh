#!/bin/bash

if [[ $# -ne 2 ]] ; then
    echo "Usage: $0 <DTYPE> <SHARDING>"
    exit 1
fi

if [[ -z "$TRTLLM_HOME" ]] ; then
    echo "TRTLLM_HOME unset"
    exit 1
fi

INPUT="$TRTLLM_HOME/context/sherlock_2000.txt"
INPUT_LEN=2000
OUTPUT_LEN=128

DTYPE=$1
SHARDING=$2

ENGINE_DIR="$ENGINES"/"$MODELNAME"_"$DTYPE"_context_"$MAX_INPUT"_"$MAX_OUTPUT"_batch_"$MAX_BATCH"_TP_"$SHARDING"

echo "Running $MODELNAME (dtype $DTYPE) with input length $INPUT_LEN, output length $OUTPUT_LEN, sharding $SHARDING."
echo "Loading engine" "$ENGINE_DIR"

if [[ "$SHARDING" == "1" ]] ; then
    time /opt/bin/cuda-reserve.py --num-gpus $SHARDING python $TRTLLM_HOME/examples/llama/run.py --max_output_len=$OUTPUT_LEN --tokenizer_dir $TOKENIZER --engine_dir $ENGINE_DIR --input_textfile $INPUT
else
    time /opt/bin/cuda-reserve.py --num-gpus $SHARDING mpirun -n $SHARDING --allow-run-as-root python $TRTLLM_HOME/examples/llama/run.py --max_output_len=$OUTPUT_LEN --tokenizer_dir $TOKENIZER --engine_dir $ENGINE_DIR --input_textfile $INPUT
fi

