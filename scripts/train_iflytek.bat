@echo off

set task="iflytek"
set data_dir="E:/donny/code/school/research/subchar-tokenization/data/%task%/split"
set lr="5e-5"
set model_name="char"
set ckpt="E:/donny/code/school/research/subchar-tokenization/checkpoints/SubChar12L20G/char/ckpt_22000.pt"
set seed=0
set test_name="./test"

@REM set output_dir="./result/%task%/char_seed%seed%_lr%lr%"

@REM Build command
set cmd=python run_glue.py
set cmd=%cmd% --lr %lr%
set cmd=%cmd% --task_name %task%
set cmd=%cmd% --train_dir %data_dir%
set cmd=%cmd% --dev_dir %data_dir%
set cmd=%cmd% --test_dir %data_dir%
set cmd=%cmd% --mode train_test
set cmd=%cmd% --init_ckpt %ckpt%
@REM set cmd=%cmd% --output_dir %output_dir%
set cmd=%cmd% --tokenizer_name %model_name%
set cmd=%cmd% --config_file configs/bert_config_vocab22675.json
set cmd=%cmd% --epochs 4
set cmd=%cmd% --seed %seed%
set cmd=%cmd% --test_name %test_name%

@REM Execute
echo "%cmd%"
%cmd%