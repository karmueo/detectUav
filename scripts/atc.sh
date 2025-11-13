#!/bin/bash

MODEL=$1
BATCH_SIZE=$2

atc --framework=5 --model=$MODEL --input_format=NCHW --input_shape="images:$BATCH_SIZE,3,640,640" --output=${MODEL%.*}_bs${BATCH_SIZE} --log=error --soc_version=Ascend310B1 --insert_op_conf=./model/aipp_nv12.cfg
