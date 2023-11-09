#!/bin/bash

if [[ $# -ne 1 ]] ; then
    echo "Please provide data type, one of int8 int4 fp8"
    exit 1
fi

DTYPE=$1

case $DTYPE in
    "int8"|"int4"|"fp8") ;;
    *) echo "Please provide data type, one of int8 int4 fp8"; exit 1 ;;
esac


TRTLLM_HOME="/code/tensorrt_llm"  # path inside container
TOKENIZER="/models/codellama/CodeLlama-34b-Instruct-hf/"  # path inside container
MODELNAME="CodeLlama-34b-instruct-hf"
ENGINES="$TRTLLM_HOME/engines"

# These values are lowered from 512 to fit on one gpu. As far as I know, it affect quality primarily (not perf). Can raise for p5d tests if needed.
CALIB_SIZE=128
BLOCK_SIZE=128

INPUT="$TRTLLM_HOME/context/sherlock_2000.txt"
INPUT_LEN=2000
OUTPUT_LEN=128

mkdir -p $ENGINES  # make sure engines parent directory exists
ENGINE_DIR="$ENGINES/$MODELNAME_$DTYPE_context_2k_128"

# Start
echo $( date )

# build step
if [[ "$DTYPE" == "fp8" ]] ; then
    echo "Building $MODELNAME in $DTYPE with calibration size $CALIB_SIZE, block size $BLOCK_SIZE, max input $INPUT_LEN, max output $OUTPUT_LEN. Writing engine to $ENGINE_DIR."
    echo "Quantizing"
    time python $TRTLLM_HOME/examples/llama/quantize.py \
         --model_dir $TOKENIZER \
         --dtype float16 \
         --qformat fp8 \
         --export_path $ENGINE_DIR \
         --calib_size $CALIB_SIZE \
         --block_size $BLOCK_SIZE

    echo "Building engine"
    time python $TRTLLM_HOME/examples/llama/build.py \
         --quantized_fp8_model_path $ENGINE_DIR/llama_float16_tp1_rank0.engine \
         --enable_fp8 \
         --fp8_kv_cache \
         --model_dir $TOKENIZER \
         --dtype float16 \
         --remove_input_padding \
         --use_gpt_attention_plugin float16 \
         --use_gemm_plugin float16 \
         --use_rmsnorm_plugin float16 \
         --enable_context_fmha \
         --output_dir $ENGINE_DIR \
         --rotary_base 1000000 \
         --vocab_size 32016 \
         --max_input_len $INPUT_LEN \
         --max_output_len $OUTPUT_LEN \
         --max_batch_size 1
else
    echo "Building $MODELNAME in $DTYPE with max input $INPUT_LEN, max output $OUTPUT_LEN. Writing engine to $ENGINE_DIR."
    time python $TRTLLM_HOME/examples/llama/build.py \
         --model_dir $TOKENIZER \
         --rotary_base 1000000 \
         --vocab_size 32016 \
         --dtype float16 \
         --remove_input_padding \
         --use_gpt_attention_plugin float16 \
         --use_gemm_plugin float16 \
         --use_rmsnorm_plugin float16 \
         --output_dir $ENGINE_DIR \
         --enable_context_fmha \
         --use_weight_only \
         --weight_only_precision $DTYPE \
         --max_input_len $INPUT_LEN \
         --max_output_len $OUTPUT_LEN \
         --max_batch_size 1
fi


# run step
echo "Running $MODELNAME (dtype $DTYPE) with input length $INPUT_LEN, output length $OUTPUT_LEN."
time python $TRTLLM_HOME/examples/llama/run.py --max_output_len=$OUTPUT_LEN --tokenizer_dir $TOKENIZER --engine_dir $ENGINE_DIR --input_textfile $INPUT

