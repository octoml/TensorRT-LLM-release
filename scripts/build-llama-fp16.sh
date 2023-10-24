export SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
export TRTLLM_HOME=$SCRIPT_DIR/..

if [[ $# -ne 2 ]] ; then
    echo "Usage scripts/build-llama-fp16.sh <model-input-dir> <model-output-dir>"
    exit 1
fi

python $TRTLLM_HOME/examples/llama/build.py --model_dir $1 --dtype float16 --use_gpt_attention_plugin float16 --use_gemm_plugin float16 --output_dir $2 --max_input_len 2364 --max_output_len 2364 --max_batch_size 1