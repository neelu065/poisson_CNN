from __future__ import print_function
import time
import numpy as np
import pylab as py
import scipy.sparse as sp  # import sparse matrix library
from scipy.sparse.linalg import spsolve

py.rcParams.update({'font.size': 14})
from diff_matrices import Diff_mat_1D, Diff_mat_2D


def diffusion_eqn_sparse(Nx, Ny, T_batch, g):
    # Dirichlet boundary conditions
    uL = T_batch[0]
    uR = T_batch[1]
    uT = T_batch[2]
    uB = T_batch[3]

    # Defining custom plotting functions
    def my_contourf(x, y, F, ttl):
        # py.contourf(x,y,F,41,cmap = 'RdBu')
        # py.contourf(F, cmap='RdBu')
        py.imshow(abs(F), cmap='RdBu', origin='lower')
        py.colorbar()
        py.title(ttl)
        py.savefig('new_save_contour' + ' ' + str(Nx) + '_' + str(Ny) + '.png', dpi=300)
        py.close()
        return 0

    x = np.linspace(-3, 3, Nx)  # x variables in 1D
    y = np.linspace(-3, 3, Ny)  # y variable in 1D

    dx = x[1] - x[0]  # grid spacing along x direction
    dy = y[1] - y[0]  # grid spacing along y direction

    X, Y = np.meshgrid(x, y)  # 2D meshgrid

    # 1D indexing
    Xu = X.ravel()  # Unravel 2D meshgrid to 1D array
    Yu = Y.ravel()

    # Source function (right hand side vector)

    # g = np.repeat(15, Nx*Ny)
    # g = np.min(uL, uR, uT, uB) + np.random.rand(Nx*Ny)
    # g = np.random.uniform(low=np.min((uL, uR, uT, uB)), high=np.max((uL, uR, uT, uB)), size=(Nx * Ny))

    # Loading finite difference matrix operators

    Dx_2d, Dy_2d, D2x_2d, D2y_2d = Diff_mat_2D(Nx, Ny)  # Calling 2D matrix operators from funciton

    # Boundary indices
    start_time = time.time()
    ind_unravel_L = np.squeeze(np.where(Xu == x[0]))  # Left boundary
    ind_unravel_R = np.squeeze(np.where(Xu == x[Nx - 1]))  # Right boundary
    ind_unravel_B = np.squeeze(np.where(Yu == y[0]))  # Bottom boundary
    ind_unravel_T = np.squeeze(np.where(Yu == y[Ny - 1]))  # Top boundary

    ind_boundary_unravel = np.squeeze(
        np.where((Xu == x[0]) | (Xu == x[Nx - 1]) | (Yu == y[0]) | (Yu == y[Ny - 1])))  # All boundary
    ind_boundary = np.where((X == x[0]) | (X == x[Nx - 1]) | (Y == y[0]) | (Y == y[Ny - 1]))  # All boundary
    print("Boundary search time = %1.6s" % (time.time() - start_time))

    # Construction of the system matrix
    start_time = time.time()
    I_sp = sp.eye(Nx * Ny).tocsr()
    L_sys = D2x_2d / dx ** 2 + D2y_2d / dy ** 2  # system matrix without boundary conditions

    L_sys[ind_boundary_unravel, :] = I_sp[ind_boundary_unravel, :]

    # Construction of right hand vector (function of x and y)
    b = g
    b[ind_unravel_L] = uL
    b[ind_unravel_R] = uR
    b[ind_unravel_T] = uT
    b[ind_unravel_B] = uB
    print("System matrix and right hand vector computation time = %1.6s" % (time.time() - start_time))

    # solve
    start_time = time.time()
    u = spsolve(L_sys, b).reshape(Ny, Nx)
    print("spsolve() time = %1.6s" % (time.time() - start_time))

    # Plot solution
    py.figure(figsize=(14, 10))
    my_contourf(x, y, u, r'$\nabla^2 u = f(x,y) OR constant$')

    return b, dx, dy, u
    # return rhs, dx, dy, actual_temp
