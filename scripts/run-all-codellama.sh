export SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
export TRTLLM_HOME=$SCRIPT_DIR/..

if [[ $# -ne 2 ]] ; then
    echo "Usage scripts/build-llama.sh <tokenizer-dir> <engine-dir>"
    exit 1
fi

$TRTLLM_HOME/scripts/run-codellama.sh 128 128 $1 $2
$TRTLLM_HOME/scripts/run-codellama.sh 128 2364 $1 $2
$TRTLLM_HOME/scripts/run-codellama.sh 2364 128 $1 $2
$TRTLLM_HOME/scripts/run-codellama.sh 2364 2364 $1 $2
