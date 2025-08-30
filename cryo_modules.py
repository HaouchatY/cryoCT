#%matplotlib inline
from tqdm.notebook import trange, tqdm
import jupyter_compare_view
import numpy as np
from jupyter_compare_view import compare
import jupyter
import matplotlib.pyplot as plt
import scipy as sp
import warnings as warn
import pyxu.experimental.xray as pxr
import sys
import mrcfile
import tqdm
import pyxu.util as pxu
import pyxu.opt.stop as pxst
import cupy as cp
import pyxu.abc as pxa
from pyxu.opt.solver import pgd, PGD, CV
from pyxu.abc import ProxFunc
from pyxu.operator import Convolve
import matplotlib
from contrasttransferfunction import ContrastTransferFunction
import torch
from pyxu.operator.interop import from_torch
from pyxu.operator import Gradient, SquaredL2Norm, L1Norm, L21Norm, PositiveL1Norm, PositiveOrthant
from pyxu.opt.solver import PD3O
from pyxu.operator import blocks
import scipy.ndimage.interpolation as interp
import time
import torch
import random
import numpy as np
import torch.nn as nn
from training.utils import utils
from training.utils.utilities import batch_PSNR
from matplotlib import pyplot as plt
import pyxu.runtime as pxrt


def build_operator(proj_filename, tlt_filename, ratio=2, roi_volume=400, width_volume=200, pitch=1.):

    # Load tilt angles from .tlt file
    with open(tlt_filename, 'r') as file:
        try :
            theta = np.array([float(line.strip()) for line in file.readlines()])
            print('Tilt angles loaded from .tlt file')
        except:
            print('Error loading tilt angles from .tlt file')
            sys.exit(1)

    # Load the MRC file
    with mrcfile.open(proj_filename, 'r') as mrc:
        try:
            tilt_series = mrc.data # (Nangles, 1024 1024)
            print('Tilt series loaded from .mrc file')
        except:
            print('Error loading tilt series from .mrc file')
            sys.exit(1)

    N_view, N_h, N_w = tilt_series.shape

    print('Creating X-rays geometry ...')
    new_tilt = np.zeros((N_view, ratio*N_h, ratio*N_w))

    for kk in range(N_view):
        new_tilt[kk] = interp.zoom(tilt_series[kk], ratio, order=0)

    tilt_series = new_tilt

    N_view, N_h, N_w = tilt_series.shape
    roi = roi_volume *ratio
    width, N_side, N_depth = width_volume * ratio, N_h, 2*roi                                                                              

    v_shape = (width_volume, N_h//ratio, 2*roi_volume)

    N_angle = tilt_series.shape[0]
    N_offset = tilt_series.shape[1]
    N_side = tilt_series.shape[1]

    # Let's build the necessary components to instantiate the operator. ========================
    angle = np.array(theta)*np.pi/180 
    n = np.stack([np.cos(angle), np.sin(angle)], axis=1)

    t = n[:, [1, 0]] * np.r_[-1, 1]

    t_max = pitch * (N_side-1) / 2 * 1 
    t_offset = np.linspace(-t_max, t_max, N_offset, endpoint=True)

    n_spec = np.broadcast_to(n.reshape(N_angle, 1, 2), (N_angle, N_offset, 2))  # (N_angle, N_offset, 2)
    t_spec = t.reshape(N_angle, 1, 2) * t_offset.reshape(N_offset, 1)  # (N_angle, N_offset, 2)

    #last dimension is the z axis depth offset. broadcast to shape (N_angle, N_offset, N_offset)
    n_spec = np.broadcast_to(n_spec.reshape(N_angle, N_offset, 1, 2), (N_angle, N_offset, N_depth, 2))
    # add new axis of n_spec is the z axis depth angle, which is 0 for all rays : shape is (N_angle, N_offset, N_offset, 3)
    n_spec = np.concatenate([n_spec, np.zeros((N_angle, N_offset, N_depth, 1))], axis=-1)

    t_spec = np.broadcast_to(t_spec.reshape(N_angle, N_offset, 1, 2), (N_angle, N_offset, N_depth, 2))
    t_spec = np.concatenate([t_spec, np.zeros((N_angle, N_offset, N_depth, 1))], axis=-1)
    t_max = pitch * (N_depth-1) / 2 * 1  # 10% over ball radius
    t_spec[:,:,:,2] = np.linspace(-t_max, t_max, N_depth, endpoint=True).reshape(1,1,N_depth) 

    N_side = int(1.5*N_side)
    v_shape = v_shape[0], int(v_shape[1]*1.5), v_shape[2]
    t_spec += pitch * np.array([1.*width/2, (1.*N_side)/2, 1*N_depth/2]) # Move anchor point to the center of volume. before 5*Nwidth/2

    print('GPU conversion ...')

    t_spec = cp.array(t_spec, dtype=cp.float32)
    n_spec = cp.array(n_spec, dtype=cp.float32)

    print('Creating operator ...')

    op_para_u = pxr.XRayTransform.init(
        arg_shape=v_shape, #volume size
        origin= (0, 0, 0),  # bottom-left corner of volume located at (0, 0) (N_side/2 - width/2, 0, 0)
        pitch=(ratio,ratio,ratio),  # pixel dimensions. (Can vary per axis.) before (5.,1.,1.)
        method="ray-trace",
        n_spec=n_spec.reshape(-1, 3),  # (N_ray, 3) directions
        t_spec=t_spec.reshape(-1, 3),  # (N_ray, 3)
    )

    print('Operator created !')
    print('volume shape : ', v_shape, '\n',
      'Operator shape : ', op_para_u.shape, '\n',
      'Geometry \n - N_views :', t_spec.shape[0], '\n - N_offsets :', t_spec.shape[1], '\n - N_depth :', t_spec.shape[2])
    
    return roi, t_spec, n_spec, v_shape, tilt_series, N_view, op_para_u

def power_iteration(A, v_shape, mempool, tol: int = 1e-1, maxiter: int = 100):
    # Ideally choose a random vector
    # To decrease the chance that our vector
    # Is orthogonal to the eigenvector
    b_k = cp.random.rand(np.prod(v_shape))
    rel_err = tol*10
    eigenval_max_old = 0

    with pxrt.Precision(pxrt.Width.SINGLE):
        for i in range(maxiter):

            # calculate the matrix-by-vector product Ab
            b_k1 = A((b_k).reshape(-1))
            # calculate the norm
            b_k1_norm = cp.linalg.norm(b_k1)

            # re normalize the vector
            b_k = b_k1 / b_k1_norm

            # free memory
            del b_k1
            mempool.free_all_blocks()

            eigenval_max = cp.dot(b_k.T, A(b_k)) / cp.dot(b_k.T, b_k) # Rayleigh quotient
            rel_err = cp.abs(eigenval_max - eigenval_max_old)
            eigenval_max_old = eigenval_max
            if i % 10 == 0:
                print('Iteration : ', i, ' | Eigenvalue : ', eigenval_max, ' | Relative error : ', rel_err)

            if rel_err < tol:

                del b_k
                mempool.free_all_blocks()
                return eigenval_max
            
    print('L_loss :', eigenval_max)

    return eigenval_max

def grad_subvolume(model, x, sigma=None, cache_wx=False, N_subvolumes=1):
            
    """ Gradient of the loss at location x. Update conv.L before if needed."""
    # first multi convolution layer
    x = (x - torch.min(x))/(torch.max(x) - torch.min(x))
    grad = torch.zeros_like(x)
    supp = 10 #hardcoded +10 to take border into account
    with torch.no_grad():
        for i in range(N_subvolumes):
            # Dimensions are (batch, channels, depth, height, width)
            if i == N_subvolumes - 1:
                supp = 0
            x_sub = x[:, :, :, i * x.shape[3] // N_subvolumes : (i + 1) * x.shape[3] // N_subvolumes + supp, :] 
            grad_sub = model.grad(x_sub, sigma=sigma)
            #print(grad[:, :, :, i * x.shape[3] // N_subvolumes : (i + 1) * x.shape[3] // N_subvolumes, :].shape)
            #print(grad_sub[:, :, :, :-supp, :].shape)
            grad[:, :, :, i * x.shape[3] // N_subvolumes : (i + 1) * x.shape[3] // N_subvolumes, :] = grad_sub[:, :, :, :grad_sub.shape[3]-supp, :]

    return(grad * (torch.max(x) - torch.min(x)) + torch.min(x))