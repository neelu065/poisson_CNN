# poisson_CNN
[![PyPI version](https://badge.fury.io/py/poisson-CNN.svg)](https://badge.fury.io/py/poisson-CNN)

poisson_CNN is a convolutional neural network model whih estimates the solution of the Poisson equation with four Dirichlet boundary conditions on rectangular grids of variable sizes.

Installation requires CUDA set up to work with tensorflow-gpu version 2.3 or newer. To install, please use the Dockerfile appropriate for your CPU architecture (in most cases, `docker/Dockerfile-amd64`)

An article describing the performance of our model is available: [journal](https://doi.org/10.1017/dce.2021.7) | [arXiv](https://arxiv.org/abs/1910.08613) 

If you use this repo in your work, please cite our paper:

Özbay AG, Hamzehloo A, Laizet S, Tzirakis P, Rizos G, Schuller B. Poisson CNN: Convolutional neural networks for the solution of the Poisson equation on a Cartesian mesh. Data-Centric Engineering. [Online] Cambridge University Press; 2021;2: e6. Available from: doi:10.1017/dce.2021.7

Horovod Installation.
1) set CUDA_HOME env variable to cuda/11.0
2) reduce the gcc module < 9
3) set LD_LIBRARY_PATH=/home/j20210241/AI_CFD/nccl/build/lib:$LD_LIBRARY_PATH
4) execute HOROVOD_NCCL_INCLUDE=/home/j20210241/AI_CFD/nccl/build/include HOROVOD_NCCL_LIB=/home/j20210241/AI_CFD/nccl/build/lib HOROVOD_GPU_OPERATIONS=NCCL pip install --no-cache-dir horovod
