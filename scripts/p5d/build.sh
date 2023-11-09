#!/bin/bash

if [[ $# -ne 2 ]] ; then
    echo "Usage: $0 <DTYPE> <SHARDING>"
    exit 1
fi

source docker-env.sh

if [[ -z "$TRTLLM_HOME" ]] ; then
    echo "TRTLLM_HOME unset"
    exit 1
fi

DTYPE=$1
SHARDING=$2

case $DTYPE in
    "int8"|"int4"|"fp8"|"fp16") ;;
    *) echo "Please provide data type, one of int8 int4 fp8 fp16"; exit 1 ;;
esac

CALIB_SIZE=128
BLOCK_SIZE=128

ENGINE_DIR="$ENGINES"/"$MODELNAME"_"$DTYPE"_context_"$MAX_INPUT"_"$MAX_OUTPUT"_batch_"$MAX_BATCH"_TP_"$SHARDING"
mkdir -p $ENGINE_DIR

if [[ -e $ENGINE_DIR ]] ; then
    echo "Engine $ENGINE_DIR already exists, skipping build."
    exit 0
fi

# Start build
if [[ "$DTYPE" == "fp8" ]] ; then
    echo "Building $MODELNAME in $DTYPE with calibration size $CALIB_SIZE, block size $BLOCK_SIZE, max input $MAX_INPUT, max output $MAX_OUTPUT, sharding $SHARDING. Writing engine to $ENGINE_DIR."

    if [[ ! -e $ENGINE_DIR/llama_tp1_rank0.npz ]] ; then
	echo "Quantizing"
	time /opt/bin/cuda-reserve.py --num-gpus 1 python $TRTLLM_HOME/examples/llama/quantize.py \
             --model_dir $TOKENIZER \
             --dtype float16 \
             --qformat fp8 \
             --export_path $ENGINE_DIR \
             --calib_size $CALIB_SIZE \
             --block_size $BLOCK_SIZE
    else
	echo "Quantized weights found, skipping quantization step."
    fi

    echo "Building engine"
    time /opt/bin/cuda-reserve.py --num-gpus $SHARDING python $TRTLLM_HOME/examples/llama/build.py \
         --quantized_fp8_model_path $ENGINE_DIR/llama_tp1_rank0.npz \
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
         --max_input_len $MAX_INPUT \
         --max_output_len $MAX_OUTPUT \
         --max_batch_size $MAX_BATCH \
	 --world_size $SHARDING \
	 --tp_size $SHARDING \
	 --parallel_build
    
else
    echo "Building $MODELNAME in $DTYPE with max input $MAX_INPUT, max output $MAX_OUTPUT, sharding $SHARDING. Writing engine to $ENGINE_DIR."
    if [[ "$DTYPE" == "fp16" ]] ; then
	QUANT_OPTS=""
    else
	QUANT_OPTS=" --use_weight_only --weight_only_precision $DTYPE"
    fi
    time /opt/bin/cuda-reserve.py --num-gpus $SHARDING python $TRTLLM_HOME/examples/llama/build.py \
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
         --max_input_len $MAX_INPUT \
         --max_output_len $MAX_OUTPUT \
         --max_batch_size $MAX_BATCH \
	 --world_size $SHARDING \
	 --tp_size $SHARDING \
	 --parallel_build $QUANT_OPTS
fi
