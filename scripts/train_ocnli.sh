#!/bin/bash

task="ocnli"

data_dir="/home/chenyingfa/subchar-tokenization/data/${task}/split"
test_dir="/home/chenyingfa/subchar-tokenization/data/${task}/split"
test_name="test"

model_name="char"

# model_name="raw"
# tok_type="RawZh"
# vocab_name="raw_zh_22675"

# model_name="pinyin"
# tok_type="CommonZh"
# vocab_name="pinyin_zh_22675"

# model_name="pinyin_no_index"
# tok_type="CommonZhNoIndex"
# vocab_name="pinyin_no_index_22675"

# model_name="pypinyin"
# tok_type="Pypinyin"
# vocab_name="pypinyin_22675_notone_noindex"

# model_name="pypinyin_12L"
# tok_type="Pypinyin"
# vocab_name="pypinyin_22675_notone_noindex"

# model_name="pypinyin_nosep_12L"
# tok_type="PypinyinNosep"
# vocab_name="pypinyin_22675_notone_noindex_nosep"

seed="0"
ckpt="/home/chenyingfa/subchar-tokenization/SubChar12L20G/char/ckpt_22000.pt"

cmd="python3 run_glue.py"
cmd+=" --task_name ${task}"
cmd+=" --train_dir=${data_dir}"
cmd+=" --dev_dir=${data_dir}"
cmd+=" --test_dir=${test_dir}"
cmd+=" --mode train_test"
cmd+=" --init_ckpt ${ckpt}"
cmd+=" --tokenizer_name ${model_name}"
cmd+=" --config_file configs/bert_config_vocab22675.json"
cmd+=" --epochs 6"
cmd+=" --lr 5e-5"
cmd+=" --seed ${seed}"
cmd+=" --test_name ${test_name}"
# cmd+=" --tokenize_char_by_char"

logfile="${output_dir}/train.log"

$cmd | tee logfile

