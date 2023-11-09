#!/bin/bash

if [[ -n "$TRTLLM_HOME" ]] ; then
    echo "TRTLLM_HOME unset"
    exit 1
fi

export cuda_version=$(nvcc --version | grep 'release' | awk '{print $6}' | awk -F'[V.]' '{print $2$3}')
export python_version=$(python3 --version 2>&1 | awk '{print $2}' | awk -F. '{print $1$2}')
wget https://developer.nvidia.com/downloads/assets/cuda/files/nvidia-ammo/nvidia_ammo-0.3.0.tar.gz
tar -xzf nvidia_ammo-0.3.0.tar.gz
pip install nvidia_ammo-0.3.0/nvidia_ammo-0.3.0+cu$cuda_version-cp$python_version-cp$python_version-linux_x86_64.whl
$TRTLLM_HOME/scripts/build_wheel.py --trt_root /usr/local/tensorrt --cuda_architectures "90-real"
pip install $TRTLLM_HOME/build/tensorrt_llm*.whl
pip install transformers==4.33.2
pip install -r $TRTLLM_HOME/examples/quantization/requirements.txt
pip install -r $TRTLLM_HOME/examples/llama/requirements.txt
rm -rf nvidia_ammo*
