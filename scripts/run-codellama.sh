export SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
export TRTLLM_HOME=$SCRIPT_DIR/..

if [[ $# -ne 4 ]] ; then
    echo "Usage scripts/build-llama.sh <seqlen> <genlen> <tokenizer-dir> <engine-dir>"
    exit 1
fi

if [ $1 -eq 128 ] ; then
    python $TRTLLM_HOME/examples/llama/run.py --max_output_len=$2 --tokenizer_dir $3 --engine_dir $4 --input_textfile $TRTLLM_HOME/scripts/renaissance_128.txt
elif [ $1 -eq 2364 ] ; then
    python $TRTLLM_HOME/examples/llama/run.py --max_output_len=$2 --tokenizer_dir $3 --engine_dir $4 --input_textfile $TRTLLM_HOME/scripts/sherlock_2364.txt
else
    echo "invalid input length"
    exit 1
fi
