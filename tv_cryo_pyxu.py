import matplotlib.pyplot as plt
import scipy as sp
#import skimage as ski
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
from pyxu.operator import Gradient, SquaredL2Norm, L1Norm, L21Norm, PositiveL1Norm
from pyxu.opt.solver import PD3O
from pyxu.operator import blocks

from weakly_convex_ridge_regularizer.models import utils
#export PYTHONPATH="weakly_convex_ridge_regularizer/"
cp.cuda.Device(0).use()

device = 'cuda:0'
fname = 'WCRR-CNN'
model = utils.load_model(fname, device=device)
print(model)
model.eval()
model.to(device)
sn_pm = model.conv_layer.spectral_norm(mode="power_method", n_steps=500)

matplotlib.use('WebAgg')

warn.filterwarnings("ignore")
plt.rcParams["image.cmap"] = "gray"

# Check the number of arguments
if len(sys.argv) != 3: 
    print("Usage: python reconstruct.py Position_12_ali.mrc Position_12.tlt")
    # Position_12_ali.mrc Position_12.tlt
    sys.exit()

# Capture command-line arguments
mrc_filename = sys.argv[1]
tlt_filename = sys.argv[2]

# Load tilt angles from .tlt file
with open(tlt_filename, 'r') as file:
    theta = cp.array([float(line.strip()) for line in file.readlines()])

# Load the MRC file

with mrcfile.open(mrc_filename, 'r') as mrc:
    tilt_series = mrc.data # (Nangles, 1024 1024)




'''
#animation of tilt series
plt.figure('tilt series')
for i in range(tilt_series.shape[0]):
    plt.imshow(tilt_series[i,:,:])
    plt.pause(0.1)
    plt.clf()
'''




#tilt_series = tilt_series[:, tilt_series.shape[2]//2 - 100 :tilt_series.shape[2]//2 + 100 , tilt_series.shape[2]//2 - 100 :tilt_series.shape[2]//2 + 100  ]



print(tilt_series.shape)
#cut off tilt series
roi = 212

N_angle = tilt_series.shape[0]
N_offset = tilt_series.shape[1]
N_side = tilt_series.shape[1]
N_depth = 2*roi
pitch = 1.

# Let's build the necessary components to instantiate the operator. ========================

angle = cp.array(theta)*cp.pi/180 
n = cp.stack([cp.cos(angle), cp.sin(angle)], axis=1)

t = n[:, [1, 0]]  # <n, t> = 0
t[:, 0] *= -1

t_max = pitch * (N_side-1) / 2 * 1  # 10% over ball radius
t_offset = cp.linspace(-t_max, t_max, N_offset, endpoint=True)

n_spec = cp.broadcast_to(n.reshape(N_angle, 1, 2), (N_angle, N_offset, 2))  # (N_angle, N_offset, 2)
t_spec = t.reshape(N_angle, 1, 2) * t_offset.reshape(N_offset, 1)  # (N_angle, N_offset, 2)

#last dimension is the z axis depth offset. broadcast to shape (N_angle, N_offset, N_offset)
n_spec = cp.broadcast_to(n_spec.reshape(N_angle, N_offset, 1, 2), (N_angle, N_offset, N_depth, 2))
# add new axis of n_spec is the z axis depth angle, which is 0 for all rays : shape is (N_angle, N_offset, N_offset, 3)
n_spec = cp.concatenate([n_spec, cp.zeros((N_angle, N_offset, N_depth, 1))], axis=-1)

width = 200 #220 found with volume ratio                                                                         

t_spec = cp.broadcast_to(t_spec.reshape(N_angle, N_offset, 1, 2), (N_angle, N_offset, N_depth, 2))
t_spec = cp.concatenate([t_spec, cp.zeros((N_angle, N_offset, N_depth, 1))], axis=-1)
t_max = pitch * (N_depth-1) / 2 * 1  # 10% over ball radius
t_spec[:,:,:,2] = cp.linspace(-t_max, t_max, N_depth, endpoint=True).reshape(1,1,N_depth) 


N_side = int(1.5*N_side)
t_spec += pitch * cp.array([5.*width/2, (1*N_side)/2, 1*N_depth/2]) # Move anchor point to the center of volume.

t_spec = cp.array(t_spec, dtype=cp.float32)
n_spec = cp.array(n_spec, dtype=cp.float32)
 
op_para_u = pxr.XRayTransform.init(
    arg_shape=(width, N_side, N_depth), #volume size
    origin= (0, 0, 0),  # bottom-left corner of volume located at (0, 0) (N_side/2 - width/2, 0, 0)
    pitch=(5., 1, 1),  # pixel dimensions. (Can vary per axis.)
    method="ray-trace",
    n_spec=n_spec.reshape(-1, 3),  # (N_ray, 3) directions
    t_spec=t_spec.reshape(-1, 3),  # (N_ray, 3)
)

full_sino_ordered = tilt_series[:, tilt_series.shape[2]//2 - roi :tilt_series.shape[2]//2 + roi , : ]
full_sino_ordered = cp.array(cp.swapaxes(full_sino_ordered, 1,2))
full_sino_ordered = cp.array(full_sino_ordered)/cp.max(full_sino_ordered)
 

# TV prior
x0=cp.array(cp.zeros((width, N_side, N_depth))).ravel()

#grad = Gradient(arg_shape=(width, N_side, N_depth), sample = [pitch, pitch, pitch], mode = 'reflect', gpu=True)
lambda_ = 0.2
#huber_norm = L1Norm(grad.shape[0]).moreau_envelope(0.01)  # We smooth the L1 norm to facilitate optimisation
#tv_prior = lambda_ * huber_norm * grad

# Positivity + L1 norm
#posL1 = 0.005 * PositiveL1Norm(width * N_side * N_depth)

class TVFunc(ProxFunc):
    r"""
    TODO
    """

    def __init__(self,
                arg_shape,
                isotropic: bool = True,
                finite_diff_kwargs: dict = dict(),
                prox_init_kwargs: dict = dict(show_progress=False),
                prox_fit_kwargs: dict = dict(),
                ):
        r"""
        Parameters
        ----------
        arg_shape: pyct.NDArrayShape
            Shape of the input array.
        isotropic: bool
            Isotropic or anisotropic TV (Default: isotropic).
        """
         
        N_dim = len(arg_shape)

        super().__init__(shape=(1, arg_shape[0]*arg_shape[1]*arg_shape[2]))
        
        self._lipschitz = cp.inf 
        self._arg_shape = arg_shape
        
        finite_diff_op = Gradient(arg_shape, sampling=[1, 1, 1], gpu=True)
        
        #finite_diff_op.estimate_lipschitz()
        #print(finite_diff_op)
        #print("finite_diff_op : ", finite_diff_op)
        print('computing lip 1')
        finite_diff_op.lipschitz= finite_diff_op.estimate_lipschitz()
        print("-------------------------", finite_diff_op.lipschitz)
        
        self._finite_diff_op = finite_diff_op
        
        if isotropic:
            self._norm = L21Norm(arg_shape=(N_dim, *arg_shape))
        
        #self._norm = L1Norm(dim=finite_diff_op.codim)
        self._prox_init_kwargs = prox_init_kwargs
        self._prox_fit_kwargs = prox_fit_kwargs

        
    def apply(self, arr):
        return self._norm(self._finite_diff_op(arr))

    def prox(self, arr, tau = 0.05):
        ls = 1 / 2 * SquaredL2Norm(dim=arr.size).argshift(arr)
        print('computing lip 2')
        ls.diff_lipschitz = ls.estimate_diff_lipschitz()
        print("-------------------", ls.diff_lipschitz)
        breakpoint()
        slv = CV(f=ls, h=tau * self._norm, K=self._finite_diff_op, verbosity = 1)
        slv.fit(x0=arr.copy(), stop_crit = pxst.RelError(eps=1e-90, var="x", f=None, norm=2, satisfy_all=True) | pxst.MaxIter(10))
        return slv.solution().reshape(arr.shape)

# Loss
v_shape = (width, N_side, N_depth)
tv_prior = TVFunc(arg_shape=v_shape)
sigma = 1
loss = (1/ (2 * sigma**2)) * SquaredL2Norm(dim=cp.array(full_sino_ordered).size).asloss(cp.array(full_sino_ordered.ravel())) * op_para_u
# Smooth part of the posterior
smooth_posterior = loss 
smooth_posterior.diff_lipschitz = 1e5

#l21 = L21Norm(arg_shape=(3, *x0.shape), l2_axis=(0, 1,2))  # Total variation prior


# Define the solver
solver = PGD(f=smooth_posterior, g = lambda_ * tv_prior, show_progress=True, verbosity=1)
#solver = PD3O(f=smooth_posterior, g = posL1, h=lambda_ * l21, K=grad, show_progress=True, verbosity=1)

# Call fit to trigger the solver
stop_crit = pxst.RelError(eps=1e-14, var="x", f=smooth_posterior, norm=2, satisfy_all=True) | pxst.MaxIter(1)

solver.fit(x0=x0.reshape(-1), stop_crit=stop_crit, acceleration=True)

recon_tv = solver.solution().squeeze()

recon_tv = recon_tv.reshape(width, N_side, N_depth)

plt.figure('TV')
plt.imshow(recon_tv.get()[width//2,N_side//2-512:N_side//2+512,:])
plt.show()
#cp.save('recon_tv.npy', recon_tv[width//4,:,:])

breakpoint()