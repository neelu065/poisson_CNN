# poisson_CNN
[![PyPI version](https://badge.fury.io/py/poisson-CNN.svg)](https://badge.fury.io/py/poisson-CNN)

poisson_CNN is a convolutional neural network model whih estimates the solution of the Poisson equation with four Dirichlet boundary conditions on rectangular grids of variable sizes.

Installation requires CUDA set up to work with tensorflow-gpu version 2.3 or newer. To install, please use the Dockerfile appropriate for your CPU architecture (in most cases, `docker/Dockerfile-amd64`)

An article describing the performance of our model is available: [journal](https://doi.org/10.1017/dce.2021.7) | [arXiv](https://arxiv.org/abs/1910.08613) 

If you use this repo in your work, please cite our paper:

Özbay AG, Hamzehloo A, Laizet S, Tzirakis P, Rizos G, Schuller B. Poisson CNN: Convolutional neural networks for the solution of the Poisson equation on a Cartesian mesh. Data-Centric Engineering. [Online] Cambridge University Press; 2021;2: e6. Available from: doi:10.1017/dce.2021.7

Command to rerun the model:

- python3 -m poisson_CNN.train.dbcnn_legacy_train ~/AI_CFD/poisson_CNN/poisson_CNN/poisson_CNN/experiments/dbcnn.json --continue_from_checkpoint /scratch/j20210241/poisson_cnn_model_test/
- pass initial_epoch=epoch number in latest chkpt file name into model.fit()