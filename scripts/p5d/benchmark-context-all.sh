#!/bin/bash

if [[ $# -ne 1 ]] ; then
    echo "Usage: $0 <DTYPE>"
    exit 1
fi

DTYPE=$1

bash benchmark-context-4x.sh $DTYPE 1
bash benchmark-context-4x.sh $DTYPE 2
bash benchmark-context-4x.sh $DTYPE 4
bash benchmark-context-4x.sh $DTYPE 8
