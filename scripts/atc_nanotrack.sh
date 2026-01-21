#!/bin/bash

BACKBONE_MODEL=${1:-model/nanotrack_backbone.onnx}
SEARCH_MODEL=${2:-model/nanotrack_backbone_search.onnx}
HEAD_MODEL=${3:-model/nanotrack_head.onnx}

# 如 ONNX 输入/输出节点名不同，请按实际名称修改 input/output 节点名
atc --framework=5 --model=${BACKBONE_MODEL} --input_format=NCHW \
    --input_shape="input:1,3,127,127" \
    --output=${BACKBONE_MODEL%.*}_bs1 --log=error --soc_version=Ascend310B1

atc --framework=5 --model=${SEARCH_MODEL} --input_format=NCHW \
    --input_shape="input:1,3,255,255" \
    --output=${SEARCH_MODEL%.*}_bs1 --log=error --soc_version=Ascend310B1

# head 模型：input1=input template, input2=search
atc --framework=5 --model=${HEAD_MODEL} --input_format=NCHW \
    --input_shape="input1:1,96,8,8;input2:1,96,16,16" \
    --output=${HEAD_MODEL%.*}_bs1 --log=error --soc_version=Ascend310B1
