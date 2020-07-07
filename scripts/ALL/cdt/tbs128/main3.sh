#!/bin/sh
#SBATCH -N 1	  # nodes requested
#SBATCH -n 1	  # tasks requested
#SBATCH --time=96:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=3000  # memory in Mb
#SBATCH --output=/home/s1771851/out.out
#SBATCH --error=/home/s1771851/error.out
#SBATCH --exclude=charles[01-10]

export CUDA_HOME=/opt/cuda-9.0.176.1/
export CUDNN_HOME=/opt/cuDNN-7.0/
export LD_LIBRARY_PATH=${CUDNN_HOME}/lib64:${CUDA_HOME}/lib64:$LD_LIBRARY_PATH
export LIBRARY_PATH=${CUDNN_HOME}/lib64:$LIBRARY_PATH
export CPATH=${CUDNN_HOME}/include:$CPATH
export PATH=${CUDA_HOME}/bin:${PATH}
export PYTHON_PATH=$PATH

source /home/s1771851/miniconda3/bin/activate
conda activate pytorch

EXECUTABLE_FILE=/home/s1771851/git/GreedyGrad/LEARN_ALL.py
LOG_DIR=/home/s1771851/logs/learn_all/lr_mom_wd/tbs128
DATASETS_DIR=/home/s1771851/Datasets/Pytorch

python ${EXECUTABLE_FILE} \
--learn_lr True \
--learn_mom True \
--learn_wd True \
--n_lrs 5 \
--n_moms 5 \
--n_wds 5 \
--dataset CIFAR10 \
--n_runs 30 \
--n_epochs_per_run 50 \
--architecture WRN-16-1 \
--activation ReLU \
--norm_type BN \
--norm_affine True \
--noRes False \
--init_type xavier \
--init_param 1 \
--init_norm_weights 1 \
--inner_lr_init 0.0 \
--inner_lr_init_type fixed \
--inner_mom_init 0.0 \
--inner_wd_init 0.0 \
--train_batch_size 128 \
--outer_lr_init 0.1 \
--outer_momentum 0.1 \
--val_batch_size 500 \
--val_source train \
--val_train_fraction 0.05 \
--val_train_overlap False \
--lr_max_percentage_change 0.1 \
--mom_max_percentage_change 0.1 \
--wd_max_percentage_change 0.1 \
--lr_max_change_thresh 4e-2 \
--mom_max_change_thresh 5e-2 \
--wd_max_change_thresh 1e-4 \
--lr_clamp_range 0 1 \
--mom_clamp_range 0 1 \
--wd_clamp_range 0 5e-3 \
--clamp_HZ True \
--clamp_HZ_range 100 \
--clamp_grads True \
--clamp_grads_range 3 \
--zero_C_lr False \
--zero_C_mom False \
--zero_C_wd False \
--retrain_from_scratch True \
--retrain_n_epochs -1 \
--datasets_path ${DATASETS_DIR} \
--log_directory_path ${LOG_DIR} \
--epoch_log_freq 5 \
--run_log_freq 1 \
--seed 3 \
--workers 2 \
--use_gpu True