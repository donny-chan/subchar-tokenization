#!/bin/bash

task="chid"
data_dir="/home/chenyingfa/subchar-tokenization/data/${task}/split"
test_dir="/home/chenyingfa/subchar-tokenization/data/${task}/split"
test_name="test"
model_dir="/home/chenyingfa/subchar-tokenization"

model_name="char"
ckpt="${model_dir}/SubChar12L20G/char/ckpt_22000.pt"
# ckpt="${model_dir}/ckpts_12l_restart/char/ckpt_18000.pt"
output_dir="result/${task}/${model_name}_12l_20g_22000"

model_name="raw"
# ckpt="${model_dir}/SubChar12L20GNew/RawZh/ckpt_19171.pt"
ckpt="${model_dir}/SubChar12L20GNew/RawZh/ckpt_24585.pt"
output_dir="result/${task}/${model_name}_12l_20g_new_24585"

# model_name="pinyin"
# ckpt="${model_dir}/SubChar12L20GNew/Pinyin/ckpt_19170.pt"
# ckpt="${model_dir}/SubChar12L20GNew/Pinyin/ckpt_24583.pt"
# ckpt="${model_dir}/ckpts_12l_restart/pinyin/ckpt_18000.pt"

# model_name="wubi"
# ckpt="${model_dir}/SubChar12L20GNew/Wubi/ckpt_19188.pt"
# ckpt="${model_dir}/SubChar12L20GNew/Wubi/ckpt_26385.pt"
# output_dir="result/${task}/${model_name}_12l_20g_new_26385"


# 20G models (restart)
model_name="char"
ckpt_step="23985"
# model_name="pinyin"
# ckpt_step="23984"
ckpt="${model_dir}/ckpts_12l_restart/${model_name}/ckpt_${ckpt_step}.pt"
output_dir="result/chid/${model_name}_12l_restart_${ckpt_step}"
seed="0"

cmd="python3 run_chid.py"
cmd+=" --data_dir=${data_dir}"
cmd+=" --train_file=${data_dir}/train.json"
cmd+=" --train_ans_file=${data_dir}/train_answer.json"
cmd+=" --dev_file=${data_dir}/dev.json"
cmd+=" --dev_ans_file=${data_dir}/dev_answer.json"
cmd+=" --mode train_test"
cmd+=" --init_ckpt ${ckpt}"
cmd+=" --output_dir ${output_dir}"
cmd+=" --tokenizer_name ${model_name}"
cmd+=" --config_file configs/bert_config_vocab22675.json"
cmd+=" --epochs 4"
cmd+=" --grad_acc_steps 12"
cmd+=" --train_batch_size 64"
cmd+=" --lr 2e-5"
cmd+=" --seed ${seed}"
# cmd+=" --test_name ${test_name}"
# cmd+=" --tokenize_char_by_char"

logfile="${output_dir}/train.log"

$cmd | tee logfile

