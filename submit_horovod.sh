#!/bin/bash
#SBATCH --job-name horovod-multinode
#SBATCH --D .
#SBATCH --output hvd_multinode_%j.output
#SBATCH --error hvd_multinode_%j.error#SBATCH --nodes=2
#SBATCH --gres=’gpu:4'
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task 40
#SBATCH --time 00:50:00module purge; module load gcc/8.3.0 cuda/10.2 cudnn/7.6.4 nccl/2.4.8 tensorrt/6.0.1 openmpi/4.0.1 atlas/3.10.3 scalapack/2.0.2 fftw/3.3.8 szip/2.1.1 ffmpeg/4.2.1 opencv/4.1.1 python/3.7.4_MLHOSTS_FLAG="-H "
for node in $(scontrol show hostnames "$SLURM_JOB_NODELIST"); do
   HOSTS_FLAG="$HOSTS_FLAG$node-ib0:$SLURM_NTASKS_PER_NODE,"
done
HOSTS_FLAG=${HOSTS_FLAG%?}horovodrun --start-timeout 120 --gloo-timeout-seconds 120 \
-np $SLURM_NTASKS $HOSTS_FLAG --network-interface ib0 --gloo \
python3.7 tf2_keras_cifar_hvd.py --epochs 10 --batch_size 512


### https://towardsdatascience.com/distributed-deep-learning-with-horovod-2d1eea004cb2