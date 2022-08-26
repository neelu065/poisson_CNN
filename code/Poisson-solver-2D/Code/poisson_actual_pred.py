from __future__ import division
import matplotlib.pyplot as plt
import poisson_CNN
import tensorflow as tf
import json
# from solve2Ddiffusionequation import diffusion_eqn
from Poisson_2D_Dirichlet_v1 import diffusion_eqn_sparse
import os
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

Nx = 750  # Along y-direction, check aspect-ratio plot.
Ny = 750

T_left = 0
T_bottom = 0
T_right = 0
T_top = 0


def predition_model_hpnn(rhs, dx, Nx, Ny):
    cfg = poisson_CNN.convert_tf_object_names(
        json.load(open(
            '/scratch/j20210241/test_poisson_CNN_folder/poisson_CNN/poisson_CNN/poisson_CNN/experiments/hpnn.json')))
    model = poisson_CNN.models.Homogeneous_Poisson_NN_Legacy(**cfg['model'])
    _ = model([tf.random.uniform((1, 1, 500, 500)), tf.random.uniform((1, 1))])
    model.compile(loss='mse', optimizer='adam')
    model.load_weights(
        '/scratch/j20210241/test_poisson_CNN_folder/poisson_CNN/poisson_CNN/hpnn_batch_2_steps_5000_epoch_7/chkpt'
        '-epoch-07.mse-0.0000')

    trhs, sf = poisson_CNN.utils.set_max_magnitude_in_batch_and_return_scaling_factors(
        rhs.reshape([1, 1] + list(rhs.shape)).astype(np.float32), 1.0)
    tdx = tf.cast(tf.constant([[dx]]), tf.float32)
    trhs = tf.cast(trhs, tf.float32)
    pred = model([trhs, tdx])
    pred = tf.abs(pred)
    pred = (((dx * (Ny - 1)) ** 2) / sf) * pred
    plt.imshow(pred[0, 0], cmap='RdBu', origin='lower')
    plt.colorbar()
    plt.savefig('predicted_hpnn' + ' ' + str(Nx) + '_' + str(Ny) + '.png', dpi=300)
    plt.close()
    return pred[0, 0]


def abs_error_hpnn(actual_temp, pred_temp_hpnn, Nx, Ny):
    abs_per_error = ((abs(actual_temp) - abs(pred_temp_hpnn)) / abs(actual_temp)) * 100
    abs_per_error = np.nan_to_num(abs_per_error)

    plt.imshow(abs_per_error, vmax = 0, vmin = 100, cmap='RdBu', origin='lower')
    plt.colorbar()
    plt.savefig('abs_error_hpnn' + ' ' + str(Nx) + '_' + str(Ny) + '.png', dpi=300)
    plt.close()


if __name__ == "__main__":
    T_batch = np.array([T_left, T_right, T_top, T_bottom])

    # if  np.all(T_batch == 0):
    b = []
    n = Nx*Ny
    for i in range(n):
        b.append(np.sin(1 - i/n))

    b = np.array(b)
    # ------ Conventional solver ------ #
    rhs, dx, dy, actual_temp = diffusion_eqn_sparse(Nx, Ny, T_batch, b)

    # ------ rhs plot ------ #
    rhs = rhs.reshape(Nx, Ny)
    plt.imshow(rhs, cmap='RdBu', origin='lower')
    plt.colorbar()
    plt.savefig('rhs' + ' ' + str(Nx) + '_' + str(Ny) + '.png', dpi=300)
    plt.close()

    # ------ Predicted solution HPNN ------ #
    # extent = (0.0, (Nx - 1) * dx, 0.0, (Ny - 1) * dy)
    pred_temp_hpnn = predition_model_hpnn(rhs, dx, Nx, Ny)

    # ----- absolute error plot ------ #
    abs_error_hpnn(actual_temp, pred_temp_hpnn, Nx, Ny)

    # else:
    #     print("dbcnn")
    #     b = np.zeros(Nx * Ny)
    #     rhs, dx, dy, actual_temp = diffusion_eqn_sparse(Nx, Ny, T_batch, b)
        # ------ Predicted solution DBCNN ----- #
        # pred_dbcnn_temp = prediction_model_dbcnn(Nx, Ny,)
        # print(pred_dbcnn_temp)

    # ---------DBCNN function--------- #
    # def prediction_model_dbcnn(Nx, Ny):
    #     cfg = poisson_CNN.convert_tf_object_names(
    #         json.load(open(
    #             '/scratch/j20210241/test_poisson_CNN_folder/poisson_CNN/poisson_CNN/poisson_CNN/experiments/dbcnn.json')))
    #     model = poisson_CNN.models.Dirichlet_BC_NN_Legacy_2(**cfg['model'])
    #     model.compile(loss='mse', optimizer='adam')
    #     model.load_weights(
    #         '/scratch/j20210241/poisson_cnn_model_test/dbcnn_legacy_train_salloc/chkpt-epoch-48.mse-0.0002')
    #
    #     bsize = int(cfg['dataset']['batch_size'] * cfg['dataset']['batches_per_epoch'])
    #     bsize = int(cfg['dataset']['batch_size'])
    #     bsize = 1
    #     # Nx = 10
    #     # Ny = 10
    #
    #     dx = tf.random.uniform((bsize, 1))
    #     # bc = tf.random.uniform((bsize, 1, Nx))
    #     bc = tf.cast(tf.random.uniform((bsize, 1, Nx)), tf.float32)
    #     x_output_resolution = tf.constant(Ny, dtype=tf.int32)
    #
    #     pred = model([bc, dx, x_output_resolution])
    #
    #     plt.imshow(pred[0, 0], cmap='RdBu', origin='lower')
    #     plt.colorbar()
    #     plt.savefig('predicted_dbcnn' + ' ' + str(Nx) + '_' + str(Ny) + '.png', dpi=300)
    #     plt.close()
    #     return pred[0, 0]