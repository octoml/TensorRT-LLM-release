#!/bin/bash

if [[ $# -ne 2 ]] ; then
    echo "Usage: $0 <DTYPE> <SHARDING>"
    exit 1
fi

DTYPE=$1
SHARDING=$2

source docker-env.sh

bash benchmark-context.sh $DTYPE $SHARDING 128 128
bash benchmark-context.sh $DTYPE $SHARDING 2364 128
bash benchmark-context.sh $DTYPE $SHARDING 128 2364
bash benchmark-context.sh $DTYPE $SHARDING 2364 2364
