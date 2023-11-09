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

RESULTS_DIR="$TRTLLM_HOME"/results
RESULTS="$RESULTS_DIR"/output_"$DTYPE"_TP_"$SHARDING"

if [[ -e $RESULTS ]] ; then
    echo "Results file $RESULTS exists, aborting"
    exit 1
fi

time ./build.sh $DTYPE $SHARDING 1>>$RESULTS 2>>$RESULTS \
    && time ./benchmark.sh $DTYPE $SHARDING 1>>$RESULTS 2>>$RESULTS
