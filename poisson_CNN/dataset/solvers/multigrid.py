import pyamg
import tensorflow as tf
import numpy as np
import copy
import itertools
from collections.abc import Iterator

try:
    import pyamgx
except:
    print('Could not import pyamgx.')

from .cholesky import poisson_RHS

def pyamgx_solve(A, b, config = None):
    '''
    Uses the (experimental) pyamgx Python bindings to the Nvidia AMGX library to solve the system Ax=b on the GPU using multigrid.

    A: CSR format sparse matrix
    b: numpy array
    config: AMGX config. See AMGX github for details.

    Outputs a numpy array containing the solution to the equation system.
    '''
    pyamgx.initialize()
    pyamgx.register_print_callback(lambda msg: print(''))
    try:#try-except block to call pyamgx.finalize() in the case an error occurs - subsequent calls with good inputs will fail otherwise
        if config is None:
            #default config copied directly from https://github.com/NVIDIA/AMGX/blob/master/core/configs/AMG_CLASSICAL_CG.json
            config = pyamgx.Config().create_from_dict({
                "config_version": 2, 
                "solver": {
                    "print_grid_stats": 1, 
                    "solver": "AMG", 
                    "print_solve_stats": 1, 
                    "presweeps": 1, 
                    "obtain_timings": 1, 
                    "max_iters": 100, 
                    "monitor_residual": 1, 
                    "convergence": "ABSOLUTE", 
                    "scope": "main", 
                    "max_levels": 100, 
                    "cycle": "CG", 
                    "tolerance": 1e-06, 
                    "norm": "L2", 
                    "postsweeps": 1
                }
            })
            
        resources = pyamgx.Resources()
        resources.create_simple(config)
        #Allocate memory for variables on GPU
        A_pyamgx = pyamgx.Matrix()
        A_pyamgx.create(resources, mode='dDDI')
        A_pyamgx.upload_CSR(A)
    
        if not isinstance(b,np.ndarray):
            b = np.array(b)
        b = b.astype(np.float64)
        b_pyamgx = pyamgx.Vector()
        b_pyamgx.create(resources, mode='dDDI')
        b_pyamgx.upload(b)

        x = pyamgx.Vector().create(resources)
        x.upload(np.zeros(b.shape,dtype=b.dtype))
        #Solve system
        solver = pyamgx.Solver()
        solver.create(resources, config)
        solver.setup(A_pyamgx)
        solver.solve(b_pyamgx, x)
        rval = x.download()
        #Cleanup to prevent GPU memory leak
        solver.destroy()
        A_pyamgx.destroy()
        b_pyamgx.destroy()
        x.destroy()
        resources.destroy()
        config.destroy()
        pyamgx.finalize()
    
        return rval
    
    except:
        pyamgx.finalize()
        raise(RuntimeError('pyamgx variable creation or solver error. See stack trace.'))

def multigrid_poisson_solve(rhses, boundaries, dx, dy = None, system_matrix = None, tol = 1e-10, solver_init_parameters = {}, solver_run_parameters = {}, use_pyamgx = False):
    '''
    Solves the Poisson equation for the given RHSes.
    
    rhses: tf.Tensor representing the RHS functions of the Poisson equation, defined across the last 2 dimensions
    boundaries: boundary conditions of the outputs; see poisson_RHS documentation
    dx: grid spacing of the outputs
    system_matrix: Sparse Poisson equation system matrix. Generated by pyamg.gallery.poisson
    tol: Tolerance of the multigrid solver.
    solver_init_parameters: dict of optional parameters to pass to pyamg.classical.ruge_stuben_solver
    solver_run_parameters: runtime parameters for the pyamg.multilevel_solver.solve method bound to the pyamg.multilevel_solver object created by pyamg.classical.ruge_stuben_solver
    use_pyamgx: if the pyamgx python interface package to the NVIDIA AMGX library is available, set to True to use the GPU to solve the equation system. (Otherwise works on the CPU)
    
    Outputs a tf.Tensor of identical shape to rhses.
    '''
    try:
        rhses = rhses.numpy()
    except:
        pass
    
    if rhses.shape[1] == 1:
        rhses = np.squeeze(rhses, axis = 1)
    
    solns = np.zeros([rhses.shape[0], 1] + [dim for dim in rhses.shape[1:]])
    rhs_vectors = poisson_RHS(rhses, boundaries, h = dx)

    if system_matrix == None:
        system_matrix = pyamg.gallery.poisson([dim-2 for dim in rhses.shape[1:]], format = 'csr')

    interior_slice = [0, Ellipsis] + [slice(1,-1) for k in rhses.shape[1:]]
    
    if use_pyamgx:
        for k in range(rhses.shape[0]):
            interior_slice[0] = k
            solns[tuple(interior_slice)] = pyamgx_solve(system_matrix, rhs_vectors[k,...]).reshape([dim-2 for dim in rhses.shape[1:]], order = 'c')

    else:
        solver = pyamg.ruge_stuben_solver(system_matrix, **solver_init_parameters)

        for k in range(rhses.shape[0]):
            interior_slice[0] = k
            solns[tuple(interior_slice)] = solver.solve(np.squeeze(rhs_vectors[k,...]), tol = tol, **solver_run_parameters).reshape([dim-2 for dim in rhses.shape[1:]], order = 'c')

    solns[...,:,-1] = boundaries['top']
    solns[...,:,0] = boundaries['bottom']
    solns[...,0,:] = boundaries['left']
    solns[...,-1,:] = boundaries['right']

    return solns
    
