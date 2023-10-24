export SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
export TRTLLM_HOME=$SCRIPT_DIR/..

$TRTLLM_HOME/scripts/build_wheel.py --trt_root /usr/local/tensorrt --cuda_architectures "90-real"
pip install ./build/tensorrt_llm*.whl
pip install transformers==4.33.2
pip install -r $TRTLLM_HOME/examples/llama/requirements.txt
