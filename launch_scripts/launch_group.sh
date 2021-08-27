#!/bin/bash

cd ../

slurm_pre="--partition t4v2,t4v1 --exclude=gpu021,gpu115 --gres gpu:1 --mem 40gb -c 4 --job-name train_group_pred --output /scratch/ssd001/home/haoran/projects/CB_WILDS/slurm_logs/%A.log"

python sweep.py launch \
    --experiment GroupClf \
    --output_dir "/scratch/hdd001/home/haoran/cxr_bias/group_predict/" \
    --slurm_pre "${slurm_pre}" \
    --command_launcher "slurm"
