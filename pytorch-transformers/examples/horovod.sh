#!/bin/sh
export TASK_NAME=MRPC 
export MAX_SEQ_LENGTH=128

#export NUM_NODES=2
#export NUM_WORKERS_PER_NODE=1

export OMP_NUM_THREADS=80
export HOROVOD_FUSION_THRESHOLD=$((128*1024*1024))
export PER_GPU_EVAL_BATCH_SIZE=64   
export PER_GPU_TRAIN_BATCH_SIZE=64 


export OUTPUT_DIR=/home/mariam/bert-finetuning-output
 

HOSTFILE=~/hostfile-clx
 
GLUE_DIR=/home/mariam/GLUE/glue_data

which $MPI



horovodrun --start-timeout 20000 -np 2 -H 10.19.242.182:1,10.19.242.219:1 --verbose  python  run_glue_hvd.py \
    --model_type bert \
    --model_name_or_path bert-large-uncased \
    --task_name $TASK_NAME \
    --do_train \
    --do_eval \
    --do_lower_case \
    --data_dir $GLUE_DIR/$TASK_NAME \
    --max_seq_length $MAX_SEQ_LENGTH \
    --per_gpu_eval_batch_size $PER_GPU_EVAL_BATCH_SIZE \
    --learning_rate 2e-5 \
    --num_train_epochs 3.0 \
    --output_dir /tmp/$TASK_NAME/np2-ppr-1-bertlarge-localhost-and-CLX-40run \
    --per_gpu_train_batch_size $PER_GPU_TRAIN_BATCH_SIZE
 
#kill -9 $(ps -eaf | grep vmstat | awk '{print $2}')
min=$(cat /tmp/vmstatlog | sed '/memory/d' | sed '/free/d' | awk -v min=9999999999 '{if($4<min){min=$4}}END{print min} ')
top=$(cat /tmp/vmstatlog | sed '/memory/d' | sed '/free/d' | head -n 1 | awk '{print $4}')
echo "Peak memory (KB):" $((top-min))
