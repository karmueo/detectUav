#!/bin/bash

MODEL=$1

atc --framework=5 --model=$MODEL --input_format=NCHW --input_shape="template:1,3,112,112;online_template:1,3,112,112;search:1,3,224,224" --output=${MODEL%.*}_bs1 --log=error --soc_version=Ascend310B1
