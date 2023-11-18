#!/bin/bash

if [[ $# -ne 1 ]] ; then
    echo "Usage: $0 <DTYPE>"
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

bash build.sh $DTYPE 1
bash build.sh $DTYPE 2
bash build.sh $DTYPE 4
bash build.sh $DTYPE 8
