export SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
export TRTLLM_HOME=$SCRIPT_DIR/..

if [[ $# -ne 4 ]] ; then
    echo "Usage scripts/build-llama-fp8.sh <calib-size> <block-size> <model-input-dir> <model-output-dir>"
    exit 1
fi

python $TRTLLM_HOME/examples/llama/quantize.py --model_dir $3 --dtype float16 --qformat fp8 --export_path $4 --calib_size $1 --block_size $2
python $TRTLLM_HOME/examples/llama/build.py --model_dir $3 --quantized_fp8_model_path $4/llama_tp1_rank0.npz --dtype float16 --use_gpt_attention_plugin float16 --use_gemm_plugin float16 --output_dir $4 --remove_input_padding --enable_fp8 --fp8_kv_cache --max_batch_size 1 --max_input_len 2364 --max_output_len 2364
