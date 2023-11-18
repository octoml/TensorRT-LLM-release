#!/bin/bash

if [[ $# -ne 4 ]] ; then
    echo "Usage: $0 <DTYPE> <SHARDING> <INPUT_LENGTH> <OUTPUT_LENGTH>"
    exit 1
fi

if [[ -z "$TRTLLM_HOME" ]] ; then
    echo "TRTLLM_HOME unset"
    exit 1
fi

DTYPE=$1
SHARDING=$2
INPUT_LEN=$3
OUTPUT_LEN=$4

case "$INPUT_LEN" in
    "128") FNAME="sherlock_128.txt" ;;
    "2364") FNAME="sherlock_2364.txt" ;;
    *) echo "Please provide input length of 128 or 2364"; exit 1 ;;
esac

INPUT="$TRTLLM_HOME"/scripts/p5d/context/"$FNAME"

ENGINE_DIR="$ENGINES"/"$MODELNAME"_"$DTYPE"_context_"$MAX_INPUT"_"$MAX_OUTPUT"_batch_"$MAX_BATCH"_TP_"$SHARDING"
RESULTS_DIR="/code/results"
RESULTS="$RESULTS_DIR"/benchmark_output_"$DTYPE"_TP_"$SHARDING"_context_"$INPUT_LEN"_"$OUTPUT_LEN"

if [[ -e $RESULTS ]] ; then
    echo "Results file $RESULTS exists, aborting"
    exit 1
fi

touch $RESULTS


echo "Running $MODELNAME (dtype $DTYPE) with input length $INPUT_LEN, output length $OUTPUT_LEN, sharding $SHARDING."
echo "Loading engine" "$ENGINE_DIR"

if [[ "$SHARDING" == "1" ]] ; then
    (time /opt/bin/cuda-reserve.py --num-gpus $SHARDING python $TRTLLM_HOME/examples/llama/run.py --max_output_len=$OUTPUT_LEN --tokenizer_dir $TOKENIZER --engine_dir $ENGINE_DIR --input_textfile $INPUT) >> $RESULTS 2>&1
else
    (time /opt/bin/cuda-reserve.py --num-gpus $SHARDING mpirun -n $SHARDING --allow-run-as-root python $TRTLLM_HOME/examples/llama/run.py --max_output_len=$OUTPUT_LEN --tokenizer_dir $TOKENIZER --engine_dir $ENGINE_DIR --input_textfile $INPUT) >> $RESULTS 2>&1
fi

echo "Finished with status $?"
