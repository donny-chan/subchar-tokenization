#!/bin/bash

# 20G models (restart)
model_name="char"
ckpt_step="23985"
model_name="pinyin"
ckpt_step="23984"
# ckpt_step="18000"
ckpt="/home/chenyingfa/subchar-tokenization/ckpts_12l_restart/${model_name}/ckpt_${ckpt_step}.pt"
output_dir="result/c3/${model_name}_12l_restart_${ckpt_step}"

cmd="python3 run_c3.py"
cmd+=" --data_dir=/home/chenyingfa/subchar-tokenization/data/c3/split"
cmd+=" --mode train_test"
cmd+=" --init_ckpt ${ckpt}"
cmd+=" --output_dir ${output_dir}"
cmd+=" --tokenizer_name ${model_name}"
cmd+=" --config_file configs/bert_config_vocab22675.json"
cmd+=" --epochs 6"
cmd+=" --train_batch_size 8"
cmd+=" --grad_acc_steps 8"
cmd+=" --lr 1e-5"
cmd+=" --seed 0"
cmd+=" --test_name test"
# cmd+=" --tokenize_char_by_char"

logfile="${output_dir}/train.log"
$cmd | tee logfile

