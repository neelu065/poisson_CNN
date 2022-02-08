#!/bin/bash
##SBATCH --export=ALL
#SBATCH --job-name=train_model
#SBATCH --nodes 1
#SBATCH --gres=gpu:1
#SBATCH -q somnathme
#SBATCH --time=72:00:00
#SBATCH --error=job.%J.err
#SBATCH --output=job.%J.out
#SBATCH --partition=gpu
#SBATCH --requeue
#SBATCH --mail-user neelu065@gmail.com

## Interactive way

source ~/.bashrc
######## load ~/.bash_modules
##module load apps/python-package/nvhpc/21.3
##module load compiler/gcc/8.3.0
##module load compiler/cuda/11.0
##module load compiler/cudnn/8.1.0


####### load ~/.bash_export
##export CUDA_HOME=/opt/ohpc/pub/apps/cuda-11.0
##LD_LIBRARY_PATH=$HOME/amgx/build:$LD_LIBRARY_PATH
##LD_PRELOAD="/opt/ohpc/pub/compiler/gcc/8.3.0/lib64/libstdc++.so.6 $LD_PRELOAD"
##export LD_LIBRARY_PATH
##export LD_PRELOAD
##export AMGX_DIR=/home/j20210241/amgx
##export TF_CPP_MIN_LOG_LEVEL=2


####### load ~/.bash_aliases
##alias cnn_env="conda activate poisson_cnn_env"

exec cnn_env


# define and create a unique scratch directory
SCRATCH_DIRECTORY=/scratch/${USER}/poisson_cnn_model_test/${SLURM_JOBID}
mkdir -p ${SCRATCH_DIRECTORY}
cd ${SCRATCH_DIRECTORY}

echo "all commands executed correctly."
##
