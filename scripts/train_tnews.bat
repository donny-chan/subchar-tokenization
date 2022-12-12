@echo off

set data_dir="E:/donny/code/school/research/subchar-tokenization/data/tnews/split"
set model_name="char"
set ckpt="E:/donny/code/school/research/subchar-tokenization/checkpoints/SubChar12L20G/char/ckpt_22000.pt"
set output_dir="./result/tnews/char_seed0"
set seed=0
set test_name="./test"

@REM Build command
set cmd=python run_glue.py
set cmd=%cmd% --task_name tnews
set cmd=%cmd% --train_dir %data_dir%
set cmd=%cmd% --dev_dir %data_dir%
set cmd=%cmd% --test_dir %data_dir%
set cmd=%cmd% --mode train_test
set cmd=%cmd% --init_ckpt %ckpt%
set cmd=%cmd% --output_dir %output_dir%
set cmd=%cmd% --tokenizer_name %model_name%
set cmd=%cmd% --config_file configs/bert_config_vocab22675.json
set cmd=%cmd% --epochs 4
set cmd=%cmd% --seed %seed%
set cmd=%cmd% --test_name %test_name%

echo "%cmd%"
%cmd%