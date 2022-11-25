#!/bin/bash

task="iflytek"

data_dir="/home/chenyingfa/subchar-tokenization/data/iflytek/split"
test_dir="/home/chenyingfa/subchar-tokenization/data/iflytek/split"
test_name="test"

model_name="char"
# model_name="raw"
# model_name="pinyin"
# model_name="pinyin_no_index"
# model_name="pypinyin"
# model_name="pypinyin_12L"
# model_name="pypinyin_nosep_12L"

seed="0"
ckpt="/home/chenyingfa/subchar-tokenization/SubChar12L20G/char/ckpt_22000.pt"
output_dir="results/${task}/${model_name}/${seed}/"
# output_dir="temp"

cmd="python3 run_glue.py"
cmd+=" --task_name ${task}"
cmd+=" --train_dir=${data_dir}"
cmd+=" --dev_dir=${data_dir}"
cmd+=" --test_dir=${test_dir}"
cmd+=" --mode train_test"
cmd+=" --init_ckpt ${ckpt}"
# cmd+=" --output_dir ${output_dir}"
cmd+=" --tokenizer_name ${model_name}"
cmd+=" --config_file configs/bert_config_vocab22675.json"
cmd+=" --epochs 8"
cmd+=" --lr 1e-4"
cmd+=" --seed ${seed}"
cmd+=" --test_name ${test_name}"
# cmd+=" --tokenize_char_by_char"

logfile="${output_dir}/train.log"

$cmd | tee logfile

