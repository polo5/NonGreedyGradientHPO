#!/bin/sh

EXECUTABLE_FILE=/home/s1771851/git/NonGreedyGradientHPO/main.py
LOG_DIR=/home/s1771851/logs/
DATASETS_DIR=/home/s1771851/Datasets/Pytorch

python ${EXECUTABLE_FILE} \
--learn_lr True \
--learn_mom True \
--learn_wd True \
--n_lrs 5 \
--n_moms 5 \
--n_wds 5 \
--dataset CIFAR10 \
--n_runs 25 \
--n_epochs_per_run 50 \
--architecture WRN-16-1 \
--init_type xavier \
--init_param 1 \
--init_norm_weights 1 \
--inner_lr_init 0.0 \
--inner_mom_init 0.0 \
--inner_wd_init 0.0 \
--train_batch_size 256 \
--outer_lr_init 0.1 \
--outer_momentum 0.1 \
--val_batch_size 500 \
--val_train_fraction 0.05 \
--val_train_overlap False \
--lr_max_percentage_change 0.1 \
--mom_max_percentage_change 0.1 \
--wd_max_percentage_change 0.1 \
--lr_max_change_thresh 2e-2 \
--mom_max_change_thresh 5e-2 \
--wd_max_change_thresh 1e-4 \
--lr_clamp_range 0 1 \
--mom_clamp_range 0 1 \
--wd_clamp_range 0 5e-3 \
--clamp_HZ True \
--clamp_HZ_range 100 \
--clamp_grads True \
--clamp_grads_range 3 \
--retrain_from_scratch True \
--retrain_n_epochs -1 \
--datasets_path ${DATASETS_DIR} \
--log_directory_path ${LOG_DIR} \
--epoch_log_freq 5 \
--run_log_freq 1 \
--seed 1 \
--workers 2 \
--use_gpu True