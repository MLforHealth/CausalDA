#!/bin/bash
# USAGE: bash launch_exp.sh {experiment_name}

cd ../

echo "Running experiment:" "${1}"

slurm_pre="--partition t4v2,t4v1 --exclude=gpu021,gpu115 --gres gpu:1 --mem 40gb -c 4 --job-name ${1} --output /scratch/ssd001/home/haoran/projects/CB_WILDS/slurm_logs/${1}_%A.log"

python sweep.py launch \
    --experiment ${1} \
    --output_dir "/scratch/ssd001/home/haoran/wilds_results/${1}/" \
    --slurm_pre "${slurm_pre}" \
    --command_launcher "slurm"
