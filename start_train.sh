#!/bin/bash
HOME="/home/shiv/git/stanford_alpaca"
conda active stanford_alpaca
# For 8 GB GPU
python ${HOME}/train.py --model_name_or_path facebook/opt-125m --data_path ${HOME}/alpaca_data.json --output_dir ${HOME}/data/output --per_device_train_batch_size 4