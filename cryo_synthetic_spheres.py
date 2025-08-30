import cupy as cp
import matplotlib.pyplot as plt
import matplotlib
from math import sqrt
import numba as nb
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
from pyxu.opt.solver import PGD, CV
from pyxu.abc import ProxFunc

import scipy.ndimage.interpolation as interp

from pyxu.operator import Gradient, SquaredL2Norm, L1Norm, L21Norm, PositiveL1Norm, PositiveOrthant
from pyxu.opt.solver import PD3O
cp.random.seed(123)
a = cp.random.random(3)
breakpoint()
matplotlib.use('WebAgg')
warn.filterwarnings("ignore")
plt.rcParams["image.cmap"] = "gray"

import numpy as np
rec = np.load('astra.npy')
k,l,m = rec.shape
plt.figure('astra 0')
plt.imshow(rec[k//2, :, :])
plt.figure('astra 1')
plt.imshow(rec[ :, l//2, :])
plt.figure('astra 2')
plt.imshow(rec[:, :, m//2])


imod_rec_sphere = 'synthetic_data/sino_synthetic_fullangles_center_recIMOD.mrc'
with mrcfile.open(imod_rec_sphere, 'r') as mrc:
    rec_imod = mrc.data # (Nangles, 1024 1024)
v_shape_bern = rec_imod.shape
plt.figure('rec_imod 0')
plt.imshow(rec_imod[v_shape_bern[0]//2,:,:], cmap='gray')
plt.figure('rec_imod 2')
plt.imshow(rec_imod[:,v_shape_bern[1]//2,:], cmap='gray')
plt.figure('rec_imod 1')
plt.imshow(rec_imod[:, :, v_shape_bern[2]//2], cmap='gray')

cp.cuda.Device(0).use()
 

tlt_filename = 'synthetic_data/sino_synthetic_fullangles_center.tlt'
mrc_filename = 'synthetic_data/sino_synthetic_fullangles_center.mrc'

# Load tilt angles from .tlt file
with open(tlt_filename, 'r') as file:
    theta = cp.array([float(line.strip()) for line in file.readlines()])
angles = theta * cp.pi / 180
# Load the MRC file
with mrcfile.open(mrc_filename, 'r') as mrc:
    tilt_series = mrc.data # (Nangles, 1024 1024)


print(len(tilt_series))

#normalize tilt series
tilt_series = tilt_series / tilt_series.max()
#tilt_series = tilt_series[6:7, :, :]
#angles = angles[6:7]
N_view, N_h, N_w = tilt_series.shape

ratio = 2
new_tilt = np.zeros((N_view, ratio*N_h, ratio*N_w))
#TEST FOR ARTIFACTS
for kk in range(N_view):
    new_tilt[kk] = interp.zoom(tilt_series[kk], ratio, order=0)  #np.resize(tilt_series[kk], (4*N_h, 4*N_w))

tilt_series = new_tilt

N_view, N_h, N_w = tilt_series.shape
width, N_side, N_depth = N_h, N_h, N_h
v_shape = (N_h//ratio, N_h//ratio, N_h//ratio)

print(tilt_series.shape)

'''
N_view, N_h, N_w = tilt_series.shape
width, N_side, N_depth = N_h, N_h, N_h
rx_pitch = (1,1)
v_pitch = (1,1,1)
v_shape = (N_h, N_h, N_h)
dws = 1

# Compute TX position (1 projection). =====================================
rx_h = rx_pitch[0] * dws * N_h  # detector height [mm]
rx_w = rx_pitch[1] * dws * N_w  # detector width  [mm]

# Compute RX position (1 projection). =====================================
rx_ll = cp.r_[-rx_w / 2 , 0, N_h ]  # detector lower-left corner

Y, X, Z = cp.meshgrid(
    (0.5 * rx_pitch[0] * dws) + cp.arange(N_h) * (rx_pitch[0] * dws),  # height (Y-axis)
    (0.5 * rx_pitch[1] * dws) + cp.arange(N_w) * (rx_pitch[1] * dws),  # width  (X-axis)
    cp.r_[0],  # depth (Z-axis)
    indexing="ij",
)

rx_pos = rx_ll + cp.concatenate([X, Y, Z], axis=-1)  # (N_h, N_w, 3)
# Compute VOL position. ===================================================
v_dim = cp.r_[v_shape] * cp.r_[v_pitch]  # volume XYZ dimensions [mm]      
v_center = cp.r_[0, rx_h / 2, 0]  # volume center                          
v_ll = v_center - v_dim / 2  # volume lower-left corner.                   

# Compute ray parameterizations (1 projection). ===========================
t_spec = rx_pos.reshape(-1,3)  # (N_h * N_w, 3)   
n_spec = cp.broadcast_to(cp.array([0.,0.,1.]), t_spec.shape)      
#t_spec = cp.reshape(rx_pos, (-1,3)) # (N_h * N_w, 3)           

# Expand n/t_spec to include all rotations. ===============================
angle = angles #+ cp.pi
R = cp.zeros(
    (N_view, 3, 3),
)

R[:, 0, 0] = cp.cos(angle)
R[:, 2, 2] = cp.cos(angle)
R[:, 2, 0] = cp.sin(angle)
R[:, 0, 2] = -cp.sin(angle)
R[:, 1, 1] = 1

#shift t_spec third dimension to correction center of rotation
n_spec = (n_spec @ R.transpose(0, 2, 1)).reshape(-1, 3)  # (N_view * N_h * N_w, 3)
t_spec = (t_spec @ R.transpose(0, 2, 1)).reshape(-1, 3)  # (N_view * N_h * N_w, 3)

op_para_u = pxr.XRayTransform.init(
    arg_shape=v_shape, #volume size
    origin=v_ll,  # bottom-left corner of volume located at (0, 0) (N_side/2 - width/2, 0, 0)
    pitch=v_pitch,  # pixel dimensions. (Can vary per axis.)
    method="ray-trace",
    n_spec=n_spec,  # (N_ray, 3) directions
    t_spec=t_spec,  # (N_ray, 3)
)

# N, K = 10, 10
# ray_idx = np.arange(500 * 500).reshape(500, 500)[::N,::K]
# fig = op_para_u.diagnostic_plot(ray_idx)
# plt.show()
# breakpoint()
'''




N_angle = tilt_series.shape[0]
N_offset = tilt_series.shape[1]
N_side = tilt_series.shape[1]
N_depth = N_offset
pitch = 1.

# Let's build the necessary components to instantiate the operator. ========================
n = cp.stack([cp.cos(angles), cp.sin(angles)], axis=1)
t = n[:, [1, 0]]  # <n, t> = 0
t[:, 0] *= -1

t_max = pitch * (N_side-1) / 2 * 1.  # 10% over ball radius
t_offset = cp.linspace(-t_max, t_max, N_offset, endpoint=True)


n_spec = cp.broadcast_to(n.reshape(N_angle, 1, 2), (N_angle, N_offset, 2))  # (N_angle, N_offset, 2)
t_spec = t.reshape(N_angle, 1, 2) * t_offset.reshape(N_offset, 1)  # (N_angle, N_offset, 2)

#last dimension is the z axis depth offset. broadcast to shape (N_angle, N_offset, N_offset)
n_spec = cp.broadcast_to(n_spec.reshape(N_angle, N_offset, 1, 2), (N_angle, N_offset, N_depth, 2))
# add new axis of n_spec is the z axis depth angle, which is 0 for all rays : shape is (N_angle, N_offset, N_offset, 3)
n_spec = cp.concatenate([n_spec, cp.zeros((N_angle, N_offset, N_depth, 1))], axis=-1)
width = N_side #200 before

t_spec = cp.broadcast_to(t_spec.reshape(N_angle, N_offset, 1, 2), (N_angle, N_offset, N_depth, 2))
t_spec = cp.concatenate([t_spec, cp.zeros((N_angle, N_offset, N_depth, 1))], axis=-1)
t_max = pitch * (N_depth-1) / 2 * 1.  # 10% over ball radius
t_spec[:,:,:,2] = cp.linspace(-t_max, t_max, N_depth, endpoint=True).reshape(1,1,N_depth) 

t_spec += pitch * cp.array([(width)/2, (N_side)/2, (N_depth)/2]) # Move anchor point to the center of volume.

'''
N_angle = tilt_series.shape[0]
N_offset = tilt_series.shape[1]
N_side = tilt_series.shape[1]
N_depth = N_offset
pitch = 1

angle = angles
n = np.stack([np.cos(angle), np.sin(angle)], axis=1)

t = n[:, [1, 0]] * np.r_[-1, 1]  # <n, t> = 0
t_max = pitch * N_side / 2 * 1.1  # 10% over ball radius
t_offset = np.linspace(-t_max, t_max, N_offset, endpoint=True)

n_spec = np.broadcast_to(n.reshape(N_angle, 1, 2), (N_angle, N_offset, 2))  # (N_angle, N_offset, 2)
t_spec = t.reshape(N_angle, 1, 2) * t_offset.reshape(N_offset, 1)  # (N_angle, N_offset, 2)
t_spec += pitch * N_side / 2  # Move anchor point to the center of volume.
'''

op_para_u = pxr.XRayTransform.init(
    arg_shape=v_shape, #volume size
    origin= (0, 0, 0),  # bottom-left corner of volume located at (0, 0) (N_side/2 - width/2, 0, 0)
    pitch=(ratio,ratio,ratio),  # pixel dimensions. (Can vary per axis.)
    method="ray-trace",
    n_spec=n_spec.reshape(-1, 3),  # (N_ray, 3) directions
    t_spec=t_spec.reshape(-1, 3),  # (N_ray, 3)
)

print('Operator set')
# plt.figure()

# x = t_spec.get().reshape(N_angle, N_offset, N_offset, 3)[25,:,:,0]
# y = t_spec.get().reshape(N_angle, N_offset, N_offset, 3)[25,:,:,2]
# print(x.shape, y.shape)
# plt.scatter(x,y, s=0.1)
# plt.show()
# breakpoint()
# N_depth -= 2

# print(t_spec.shape)
# print(n_spec.shape)

#full sino ordered is tilt_series but index 1 and 2 are swapped
#full_sino_ordered = cp.array(cp.swapaxes(tilt_series, 1, 2))
full_sino_ordered = cp.array(tilt_series)

#convert cp arrays:
I_BP = op_para_u.adjoint(full_sino_ordered.reshape(-1)).reshape(v_shape)
I_BP = I_BP.get()
plt.figure('BP')
plt.imshow(I_BP[k//2,:,:], cmap='gray')

plt.figure('BP2')
plt.imshow(I_BP[:,l//2, :], cmap='gray')

plt.figure('BP3')
plt.imshow(I_BP[:,:, m//2], cmap='gray')

fwd = op_para_u.apply(cp.array(I_BP.reshape(-1))).reshape(full_sino_ordered.shape)
fwd = fwd.get()

stop_crit = pxst.MaxIter(10)
recon_pinv = op_para_u.pinv(full_sino_ordered.reshape(-1), damp=1, kwargs_fit=dict(stop_crit=stop_crit)).reshape(v_shape)
recon_pinv= recon_pinv.get()

plt.figure('pinv')

plt.imshow(recon_pinv[v_shape[0]//2,:,:])
plt.figure()
plt.imshow(recon_pinv[:,v_shape[1]//2, :])
plt.figure()
plt.imshow(recon_pinv[:,:, v_shape[2]//2])

plt.figure('pyxu - imod')

slice_pyxu = recon_pinv[:,:, v_shape[2]//2] / np.max(recon_pinv[:,:, v_shape[2]//2])
slice_imod = rec_imod[v_shape_bern[0]//2,:,:] / np.max(rec_imod[v_shape_bern[0]//2,:,:])

plt.imshow(slice_pyxu - slice_imod)
plt.show()

breakpoint()



# TEST WITH reg norm 1 of grad =======================================
sigma = 1
loss = (1/ (2 * sigma**2)) * SquaredL2Norm(dim=cp.array(full_sino_ordered).size).asloss(cp.array(full_sino_ordered.ravel())) * op_para_u

lambda_ = 30.

l21 = L21Norm(arg_shape=(3, *(width, N_side, N_depth)))  # Total variation prior
 
grad = Gradient(
    arg_shape=(width, N_side, N_depth),
    diff_method="fd",
    scheme="central",
    accuracy=8,
    gpu = True,
) #Gradient operator

loss = loss + lambda_ * SquaredL2Norm(dim=(3 * width * N_side * N_depth)) * grad
stop_crit = pxst.AbsError(
    eps=1e-5,
    var="x",
    f=loss,
    norm=2,
    satisfy_all=True,
)| pxst.MaxIter(50)

loss.diff_lipschitz = 1e6
positivity = PositiveOrthant(dim=(width* N_side* N_depth))
#solver = PD3O(f=loss, g = None, h=lambda_ * l21, K=grad, beta = 2e2, show_progress=True, verbosity = 1) #before beta = 1e5
solver = PGD(f=loss, g = positivity, show_progress=True, verbosity = 1) #before beta = 1e5
solver.fit(x0=cp.array(I_BP).reshape(-1), stop_crit=stop_crit)  # Solving the optimization problem
recon_pd = solver.solution().squeeze()
recon_pd = recon_pd.reshape(width, N_side, N_depth)
recon_pd= recon_pd.get()
plt.figure('pd30')

plt.imshow(recon_pd[width//2,:,:])
plt.figure()
plt.imshow(recon_pd[:,width//2, :])
plt.figure()
plt.imshow(recon_pd[:,:, width//2])

plt.show()

breakpoint()












breakpoint()

fwd = op_para_u.apply(cp.array(recon_pinv.reshape(-1))).reshape(full_sino_ordered.shape)
fwd = fwd.get()

proj = full_sino_ordered.get()
plt.figure('fwd1')
plt.imshow(fwd[len(angles)//2,:,:])
plt.figure('proj1')
plt.imshow(proj[len(angles)//2,:,:])

plt.figure('fwd2')
plt.imshow(fwd[:,width//2, :])
plt.figure('proj2')
plt.imshow(proj[:,width//2, :])

plt.figure('fwd3')
plt.imshow(fwd[:,:, width//2])
plt.figure('proj3')
plt.imshow(proj[:,:, width//2])

plt.show()


breakpoint()


#im = I_BP[I_BP.shape[0]//2, :, :]

cp.save("reconGPU_corr.npy", recon_pinv[width//2-10,:,:])



plt.figure('sino data')
#imshow with range of values for better contrast 
plt.imshow(full_sino_ordered[29,:,:])
plt.figure('BP')
plt.imshow(im)

plt.figure('recon_pinv')
plt.imshow(recon_pinv[N_side//2,:,:])


fwd = op_para_u.apply(cp.array(recon_pinv.reshape(-1))).reshape(full_sino_ordered.shape)
plt.figure('fwd')
plt.imshow(fwd[29,:,:])
plt.show()



#Try with TV prior



# TV prior
x0=cp.array(0*cp.zeros((width, N_side, N_depth))).ravel()

#grad = Gradient(arg_shape=(width, N_side, N_depth), sample = [pitch, pitch, pitch], mode = 'reflect', gpu=True)
lambda_ = 1
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
        
        finite_diff_op = Gradient(arg_shape)#, gpu=True)
        
        #finite_diff_op.estimate_lipschitz()
        #print(finite_diff_op)
        #print("finite_diff_op : ", finite_diff_op)
        finite_diff_op.lipschitz= 3
        self._finite_diff_op = finite_diff_op
        
        if isotropic:
            self._norm = L21Norm(arg_shape=(N_dim, *arg_shape))
        
        #self._norm = L1Norm(dim=finite_diff_op.codim)
        self._prox_init_kwargs = prox_init_kwargs
        self._prox_fit_kwargs = prox_fit_kwargs

        

    def apply(self, arr):
        return cp.array(self._norm(self._finite_diff_op(cp.array(arr))))

    def prox(self, arr, tau = 0.05):
        ls = 1 / 2 * SquaredL2Norm(dim=arr.size).argshift(-cp.array(arr))
        #ls.estimate_diff_lipschitz()
        ls.diff_lipschitz = 1.0
        slv = CV(f=ls, h=tau * self._norm, K=self._finite_diff_op, verbosity = 5)
        slv.fit(x0=cp.array(arr.copy()), stop_crit = pxst.MaxIter(30))
        return cp.array(slv.solution().reshape(arr.shape))

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
stop_crit = pxst.AbsError(eps=1e-4, var="x", f=smooth_posterior, norm=2, satisfy_all=True) | pxst.MaxIter(30)

solver.fit(x0=x0, stop_crit=stop_crit)

recon_tv = solver.solution().squeeze()

recon_tv = recon_tv.reshape(width, N_side, N_depth)

print('SNR')
mu = cp.mean(recon_tv)
sigma = cp.std(recon_tv)
print("SNR : ", mu/sigma)

plt.figure('tvrec')
plt.imshow(recon_tv[N_side//2,:,:])
plt.figure()
plt.imshow(recon_tv[:,width//2, :])
plt.figure()
plt.imshow(recon_tv[:,:, width//2])
plt.show()

breakpoint()

plt.figure('recon_tv')
plt.imshow(recon_tv[N_side//2,:,:])
plt.show()

breakpoint()