#!/bin/bash

TRTLLM_HOME="/code/tensorrt_llm"
STATS="$TRTLLM_HOME/stats"
STATFILE="$STATS/stats.txt"

mkdir -p $STATS
touch $STATFILE

#DTYPES=( int4 int8 fp8 )
DTYPES=( fp8 )

for DTYPE in "${DTYPES[@]}"
do
    FNAME="$STATS/output_$DTYPE"
    echo "Running $DTYPE, writing output to $FNAME"
    bash $TRTLLM_HOME/scripts/otherside/run-one.sh $DTYPE > $FNAME
    echo $DTYPE >> $STATFILE
    grep -A 1 "seqlen" $FNAME >> $STATFILE
done

