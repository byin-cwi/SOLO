#!/bin/bash
# run_pipep_8gpu.sh: run_pipep_multi.sh with eight-GPU defaults (GPUS=0..7, PS="2 8").
# Usage: nohup bash run_pipep_8gpu.sh > pipep_8gpu.log 2>&1 &
cd "$(dirname "$0")"
GPUS=${GPUS:-0,1,2,3,4,5,6,7} PS=${PS:-"2 8"} exec bash run_pipep_multi.sh
