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
from pyxu.operator import Gradient, SquaredL2Norm, L1Norm, L21Norm, PositiveL1Norm, PositiveOrthant
from pyxu.opt.solver import PD3O
from pyxu.operator import blocks

import scipy.ndimage.interpolation as interp

from weakly_convex_ridge_regularizer.models import utils

breakpoint()
#export PYTHONPATH="weakly_convex_ridge_regularizer/"
cp.cuda.Device(0).use()

'''device = 'cuda:0'
fname = 'WCRR-CNN'
model = utils.load_model(fname, device=device)
print(model)
model.eval()
model.to(device)
sn_pm = model.conv_layer.spectral_norm(mode="power_method", n_steps=500)'''

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

N_view, N_h, N_w = tilt_series.shape

ratio = 2
import numpy as np
new_tilt = np.zeros((N_view, ratio*N_h, ratio*N_w))
#TEST FOR ARTIFACTS
for kk in range(N_view):
    new_tilt[kk] = interp.zoom(tilt_series[kk], ratio, order=0)  #np.resize(tilt_series[kk], (4*N_h, 4*N_w))

tilt_series = new_tilt

N_view, N_h, N_w = tilt_series.shape
roi_volume = 200 #400 fits with pinc
width_volume = 200 # width=220 found with volume ratio   
roi = roi_volume *ratio
width, N_side, N_depth = width_volume * ratio, N_h, 2*roi                                                                              

v_shape = (width_volume, N_h//ratio, 2*roi_volume)

'''
#animation of tilt series
plt.figure('tilt series')
for i in range(tilt_series.shape[0]):
    plt.imshow(tilt_series[i,:,:])
    plt.pause(0.1)
    plt.clf()
'''

print(tilt_series.shape)
#cut off tilt series

N_angle = tilt_series.shape[0]
N_offset = tilt_series.shape[1]
N_side = tilt_series.shape[1]

pitch = 1.

# Let's build the necessary components to instantiate the operator. ========================
angle = cp.array(theta)*cp.pi/180 
n = cp.stack([cp.cos(angle), cp.sin(angle)], axis=1)

t = n[:, [1, 0]] * cp.r_[-1, 1]

t_max = pitch * (N_side-1) / 2 * 1  # 10% over ball radius
t_offset = cp.linspace(-t_max, t_max, N_offset, endpoint=True)

n_spec = cp.broadcast_to(n.reshape(N_angle, 1, 2), (N_angle, N_offset, 2))  # (N_angle, N_offset, 2)
t_spec = t.reshape(N_angle, 1, 2) * t_offset.reshape(N_offset, 1)  # (N_angle, N_offset, 2)

#last dimension is the z axis depth offset. broadcast to shape (N_angle, N_offset, N_offset)
n_spec = cp.broadcast_to(n_spec.reshape(N_angle, N_offset, 1, 2), (N_angle, N_offset, N_depth, 2))
# add new axis of n_spec is the z axis depth angle, which is 0 for all rays : shape is (N_angle, N_offset, N_offset, 3)
n_spec = cp.concatenate([n_spec, cp.zeros((N_angle, N_offset, N_depth, 1))], axis=-1)

t_spec = cp.broadcast_to(t_spec.reshape(N_angle, N_offset, 1, 2), (N_angle, N_offset, N_depth, 2))
t_spec = cp.concatenate([t_spec, cp.zeros((N_angle, N_offset, N_depth, 1))], axis=-1)
t_max = pitch * (N_depth-1) / 2 * 1  # 10% over ball radius
t_spec[:,:,:,2] = cp.linspace(-t_max, t_max, N_depth, endpoint=True).reshape(1,1,N_depth) 


N_side = int(1.5*N_side)
v_shape = v_shape[0], int(v_shape[1]*1.5), v_shape[2]
t_spec += pitch * cp.array([1.*width/2, (1.*N_side)/2, 1*N_depth/2]) # Move anchor point to the center of volume. before 5*Nwidth/2

t_spec = cp.array(t_spec, dtype=cp.float32)
n_spec = cp.array(n_spec, dtype=cp.float32)


op_para_u = pxr.XRayTransform.init(
    arg_shape=v_shape, #volume size
    origin= (0, 0, 0),  # bottom-left corner of volume located at (0, 0) (N_side/2 - width/2, 0, 0)
    pitch=(ratio,ratio,ratio),  # pixel dimensions. (Can vary per axis.) before (5.,1.,1.)
    method="ray-trace",
    n_spec=n_spec.reshape(-1, 3),  # (N_ray, 3) directions
    t_spec=t_spec.reshape(-1, 3),  # (N_ray, 3)
)

print(t_spec.shape)
print(n_spec.shape)

#full sino ordered is tilt_series but index 1 and 2 are swapped
full_sino_ordered = tilt_series[:, tilt_series.shape[2]//2 - roi :tilt_series.shape[2]//2 + roi , : ]
full_sino_ordered = cp.array(cp.swapaxes(full_sino_ordered, 1,2))
full_sino_ordered = cp.array(full_sino_ordered)/cp.max(full_sino_ordered)
 

# ========= Adjoint =========================
I_BP = op_para_u.adjoint(full_sino_ordered.reshape(-1)).reshape(v_shape)
I_BP = I_BP.get()
plt.figure('BP')
plt.imshow(I_BP[v_shape[0]//2,v_shape[1]//2-512:v_shape[1]//2+512,:], cmap='gray')
# ==========================================


# ========= IMOD reconstruction from Bern =========
rec_bern_file = 'Position_12_rec.mrc'
with mrcfile.open(rec_bern_file, 'r') as mrc:
    rec_bern = mrc.data # (Nangles, 1024 1024)
v_shape_bern = rec_bern.shape
plt.figure('recon_bern 0')
plt.imshow(rec_bern[v_shape_bern[0]//2,:,:], cmap='gray')
plt.figure('recon_bern_slice 2')
plt.imshow(rec_bern[:,v_shape_bern[1]//2,:], cmap='gray')
plt.figure('recon_bern_slice 1')
plt.imshow(rec_bern[:, :, v_shape_bern[2]//2], cmap='gray')
# =================================================



# ========= CGD FOR PSEUDO INVERSERSE REC=========================
stop_crit = pxst.MaxIter(5) #500
recon_pinv = op_para_u.pinv(cp.array(full_sino_ordered.reshape(-1)), damp=1, kwargs_fit=dict(stop_crit=stop_crit)).reshape(v_shape)
recon_pinv = recon_pinv.get()
plt.figure('recon_pinv 0')
plt.imshow(recon_pinv[v_shape[0]//2,v_shape[1]//2-512:v_shape[1]//2+512,:], cmap='gray')
plt.figure('recon_pinv_slice 2')
plt.imshow(recon_pinv[:,v_shape[1]//2-512:v_shape[1]//2+512,v_shape[2]//2], cmap='gray')
plt.figure('recon_pinv_slice 1')
plt.imshow(recon_pinv[:,v_shape[1]//2,:], cmap='gray')
# ===================================================================




# ================ GD with smoothed TV prior ==============================
sigma = 1
loss = (1/ (2 * sigma**2)) * SquaredL2Norm(dim=cp.array(full_sino_ordered).size).asloss(cp.array(full_sino_ordered.ravel())) * op_para_u

lambda_tv = 80000. #   100000. for tv hubert       10000. with ||grad x||_2
lambda_grad2 = 1000

grad = Gradient(
    arg_shape=v_shape,
    diff_method="fd",
    scheme="central",
    accuracy=8,
    gpu = True,
    sampling = [1, 1, 1]
) #Gradient operator

#l21 = L21Norm(arg_shape=(3, *v_shape))  # in case we need it ... Total variation prior

# -------- for ||grad x||_1 ---------
huber_norm = L1Norm(grad.shape[0]).moreau_envelope(0.01)
loss = loss + lambda_tv * huber_norm * grad + lambda_grad2 * SquaredL2Norm(dim=(3 * v_shape[0]*v_shape[1]*v_shape[2])) * grad

# -------- for ||grad x||_2 ---------
#loss = loss + lambda_ * SquaredL2Norm(dim=(3 * v_shape[0]*v_shape[1]*v_shape[2])) * grad
stop_crit = pxst.RelError(
    eps=1e-5,
    var="x",
    f=loss,
    norm=2,
    satisfy_all=True,
)| pxst.MaxIter(150) #200 15 before

loss.diff_lipschitz = 2e7 #2e7 1e6 with ||grad x||_2         9e6 for tv hubert etc
positivity = PositiveOrthant(dim=v_shape[0]*v_shape[1]*v_shape[2])
#solver = PD3O(f=loss, g = None, h=lambda_ * l21, K=grad, beta = 2e2, show_progress=True, verbosity = 1) #before beta = 1e5
solver = PGD(f=loss, g = None, show_progress=True, verbosity = 1) #before beta = 1e5
solver.fit(x0=cp.array(I_BP).reshape(-1), stop_crit=stop_crit)  # Solving the optimization problem
recon_pd = solver.solution().squeeze()
recon_pd = recon_pd.reshape(v_shape)
recon_pd= recon_pd.get()

plt.figure('TV_slice 0')
plt.imshow(recon_pd[v_shape[0]//2,v_shape[1]//2-512:v_shape[1]//2+512,:], cmap='gray')
plt.figure('TV_slice 2')
plt.imshow(recon_pd[:,v_shape[1]//2-512:v_shape[1]//2+512,v_shape[2]//2], cmap='gray')
plt.figure('TV_slice 1')
plt.imshow(recon_pd[:,v_shape[1]//2,:], cmap='gray')
# ================================================================================


plt.show()
breakpoint()







stop_crit = pxst.AbsError(
    eps=1e-5,
    var="x",
    f=None,
    norm=2,
    satisfy_all=True,
)| pxst.MaxIter(15)


positivity = PositiveOrthant(dim=(width* N_side* N_depth))
solver = PD3O(f=loss, g = None, h=lambda_ * l21, K=grad, beta = beta_, show_progress=True, verbosity = 1)
solver.fit(x0=cp.array(I_BP).reshape(-1), stop_crit=stop_crit)  # Solving the optimization problem
recon_pd = solver.solution().squeeze()
recon_pd = recon_pd.reshape(width, N_side, N_depth)
recon_pd= recon_pd.get()
plt.figure('pd30 15')

plt.imshow(recon_pd[width//2,N_side//2-512:N_side//2+512,:])




stop_crit = pxst.AbsError(
    eps=1e-5,
    var="x",
    f=None,
    norm=2,
    satisfy_all=True,
)| pxst.MaxIter(50)


positivity = PositiveOrthant(dim=(width* N_side* N_depth))
solver = PD3O(f=loss, g = None, h=lambda_ * l21, K=grad, beta = beta_, show_progress=True, verbosity = 1)
solver.fit(x0=cp.array(I_BP).reshape(-1), stop_crit=stop_crit)  # Solving the optimization problem
recon_pd = solver.solution().squeeze()
recon_pd = recon_pd.reshape(width, N_side, N_depth)
recon_pd= recon_pd.get()

plt.figure('pd30 50')
plt.imshow(recon_pd[width//2,N_side//2-512:N_side//2+512,:])
plt.show()

breakpoint()

'''
# TV prior
x0=cp.array(cp.zeros((width, N_side, N_depth))).ravel()
v_shape = (width, N_side, N_depth)
grad = Gradient(v_shape, sampling=pitch, gpu=True)
lambda_ = 5
huber_norm = L1Norm(grad.shape[0]).moreau_envelope(0.01)  # We smooth the L1 norm to facilitate optimisation
tv_prior = lambda_ * huber_norm * grad
tv_prior.diff_lipschitz = lambda_

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
        finite_diff_op.lipschitz= 1e7 # finite_diff_op.estimate_lipschitz()
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
        
        ls.diff_lipschitz = 1e7 # ls.estimate_diff_lipschitz()
        print("-------------------", ls.diff_lipschitz)
        slv = CV(f=ls, h=tau * self._norm, K=self._finite_diff_op, verbosity = 1)
        slv.fit(x0=arr.copy(), stop_crit = pxst.RelError(eps=1e-90, var="x", f=None, norm=2, satisfy_all=True) | pxst.MaxIter(20))
        return slv.solution().reshape(arr.shape)

# Loss

#tv_prior = TVFunc(arg_shape=v_shape)
sigma = 10
loss = (1/ (2 * sigma**2)) * SquaredL2Norm(dim=cp.array(full_sino_ordered).size).asloss(cp.array(full_sino_ordered.ravel())) * op_para_u
# Smooth part of the posterior
smooth_posterior = loss 
smooth_posterior.diff_lipschitz = 4e4

#l21 = L21Norm(arg_shape=(3, *x0.shape), l2_axis=(0, 1,2))  # Total variation prior


# Define the solver
solver = PGD(f=smooth_posterior + lambda_ * tv_prior , show_progress=True, verbosity=1)
#solver = PD3O(f=smooth_posterior, g = posL1, h=lambda_ * l21, K=grad, show_progress=True, verbosity=1)

# Call fit to trigger the solver
stop_crit = pxst.RelError(eps=1e-14, var="x", f=None, norm=2, satisfy_all=True) | pxst.MaxIter(400)

solver.fit(x0=x0.reshape(-1), stop_crit=stop_crit, acceleration=True)

recon_tv = solver.solution().squeeze()

recon_tv = recon_tv.reshape(width, N_side, N_depth)

plt.figure('TV')
plt.imshow(recon_tv.get()[width//2,N_side//2-512:N_side//2+512,:])
plt.show()
#cp.save('recon_tv.npy', recon_tv[width//4,:,:])

breakpoint()'''
#----------------trying PSF here

import numpy as np
import scipy.fft as fft
import scipy.fftpack as fftpack
import scipy.signal as signal
 
def contrast_normalization(arr_bin, tile_size = 128):
    '''
    Computes the minimum and maximum contrast values to use
    by calculating the median of the 2nd/98th percentiles
    of the mic split up into tile_size * tile_size patches.
    :param arr_bin: the micrograph represented as a numpy array
    :type arr_bin: list
    :param tile_size: the size of the patch to split the mic by 
        (larger is faster)
    :type tile_size: int
    '''
    ny,nx = arr_bin.shape
    # set up start and end indexes to make looping code readable
    tile_start_x = np.arange(0, nx, tile_size)
    tile_end_x = tile_start_x + tile_size
    tile_start_y = np.arange(0, ny, tile_size)
    tile_end_y = tile_start_y + tile_size
    num_tile_x = len(tile_start_x)
    num_tile_y = len(tile_start_y)
    
    # initialize array that will hold percentiles of all patches
    tile_all_data = np.empty((num_tile_y*num_tile_x, 2), dtype=np.float32)

    index = 0
    for y in range(num_tile_y):
        for x in range(num_tile_x):
            # cut out a patch of the mic
            arr_tile = arr_bin[tile_start_y[y]:tile_end_y[y], tile_start_x[x]:tile_end_x[x]]
            # store 2nd and 98th percentile values
            tile_all_data[index:,0] = np.percentile(arr_tile, 98)
            tile_all_data[index:,1] = np.percentile(arr_tile, 2)
            index += 1

    # calc median of non-NaN percentile values
    all_tiles_98_median = np.nanmedian(tile_all_data[:,0])
    all_tiles_2_median = np.nanmedian(tile_all_data[:,1])
    vmid = 0.5*(all_tiles_2_median+all_tiles_98_median)
    vrange = abs(all_tiles_2_median-all_tiles_98_median)
    extend = 1.5
    # extend vmin and vmax enough to not include outliers
    vmin = vmid - extend*0.5*vrange
    vmax = vmid + extend*0.5*vrange

    return vmin, vmax

# Since CTF is the Fourier transform of PSF, we can compute PSF by inverse Fourier transform of CTF


proj = full_sino_ordered[29].get()
proj /= np.max(proj)

mrcfile.write('proj.mrc', np.array(proj, dtype=np.float32), overwrite=True)

plt.figure('projection')
plt.imshow(proj)

freqs = [0.000000,0.000421,0.000842,0.001263,0.001684,0.002105,0.002526,0.002947,0.003367,0.003788,0.004209,0.004630,0.005051,0.005472,0.005893,0.006314,0.006735,0.007156,0.007577,0.007998,0.008419,0.008840,0.009261,0.009681,0.010102,0.010523,0.010944,0.011365,0.011786,0.012207,0.012628,0.013049,0.013470,0.013891,0.014312,0.014733,0.015154,0.015574,0.015995,0.016416,0.016837,0.017258,0.017679,0.018100,0.018521,0.018942,0.019363,0.019784,0.020205,0.020626,0.021047,0.021468,0.021888,0.022309,0.022730,0.023151,0.023572,0.023993,0.024414,0.024835,0.025256,0.025677,0.026098,0.026519,0.026940,0.027361,0.027782,0.028202,0.028623,0.029044,0.029465,0.029886,0.030307,0.030728,0.031149,0.031570,0.031991,0.032412,0.032833,0.033254,0.033675,0.034096,0.034516,0.034937,0.035358,0.035779,0.036200,0.036621,0.037042,0.037463,0.037884,0.038305,0.038726,0.039147,0.039568,0.039989,0.040409,0.040830,0.041251,0.041672,0.042093,0.042514,0.042935,0.043356,0.043777,0.044198,0.044619,0.045040,0.045461,0.045882,0.046303,0.046723,0.047144,0.047565,0.047986,0.048407,0.048828,0.049249,0.049670,0.050091,0.050512,0.050933,0.051354,0.051775,0.052196,0.052617,0.053037,0.053458,0.053879,0.054300,0.054721,0.055142,0.055563,0.055984,0.056405,0.056826,0.057247,0.057668,0.058089,0.058510,0.058930,0.059351,0.059772,0.060193,0.060614,0.061035,0.061456,0.061877,0.062298,0.062719,0.063140,0.063561,0.063982,0.064403,0.064824,0.065244,0.065665,0.066086,0.066507,0.066928,0.067349,0.067770,0.068191,0.068612,0.069033,0.069454,0.069875,0.070296,0.070717,0.071138,0.071558,0.071979,0.072400,0.072821,0.073242,0.073663,0.074084,0.074505,0.074926,0.075347,0.075768,0.076189,0.076610,0.077031,0.077452,0.077872,0.078293,0.078714,0.079135,0.079556,0.079977,0.080398,0.080819,0.081240,0.081661,0.082082,0.082503,0.082924,0.083345,0.083765,0.084186,0.084607,0.085028,0.085449,0.085870,0.086291,0.086712,0.087133,0.087554,0.087975,0.088396,0.088817,0.089238,0.089659,0.090079,0.090500,0.090921,0.091342,0.091763,0.092184,0.092605,0.093026,0.093447,0.093868,0.094289,0.094710,0.095131,0.095552,0.095973,0.096393,0.096814,0.097235,0.097656,0.098077,0.098498,0.098919,0.099340,0.099761,0.100182,0.100603,0.101024,0.101445,0.101866,0.102287,0.102707,0.103128,0.103549,0.103970,0.104391,0.104812,0.105233,0.105654,0.106075,0.106496,0.106917,0.107338,0.107759,0.108180,0.108600,0.109021,0.109442,0.109863,0.110284,0.110705,0.111126,0.111547,0.111968,0.112389,0.112810,0.113231,0.113652,0.114073,0.114494,0.114914,0.115335,0.115756,0.116177,0.116598,0.117019,0.117440,0.117861,0.118282,0.118703,0.119124,0.119545,0.119966,0.120387,0.120808,0.121228,0.121649,0.122070,0.122491,0.122912,0.123333,0.123754,0.124175,0.124596,0.125017,0.125438,0.125859,0.126280,0.126701,0.127122,0.127542,0.127963,0.128384,0.128805,0.129226,0.129647,0.130068,0.130489,0.130910,0.131331,0.131752,0.132173,0.132594,0.133015,0.133435,0.133856,0.134277,0.134698,0.135119,0.135540,0.135961,0.136382,0.136803,0.137224,0.137645,0.138066,0.138487,0.138908,0.139329,0.139749,0.140170,0.140591,0.141012,0.141433,0.141854,0.142275,0.142696,0.143117,0.143538,0.143959,0.144380,0.144801,0.145222,0.145643,0.146063,0.146484,0.146905,0.147326,0.147747,0.148168,0.148589,0.149010,0.149431,0.149852,0.150273,0.150694,0.151115,0.151536,0.151956,0.152377]

proj_fft = np.fft.fftshift(np.fft.fft2(proj))

proj_fft_abs2 = np.log(np.abs(proj_fft)**2)

vmin, vmax = contrast_normalization(proj_fft_abs2, 128)

CTF = [0.070000,0.071082,0.074326,0.079732,0.087297,0.097015,0.108880,0.122882,0.139008,0.157240,0.177552,0.199915,0.224289,0.250623,0.278856,0.308913,0.340705,0.374125,0.409045,0.445319,0.482776,0.521223,0.560436,0.600168,0.640138,0.680037,0.719525,0.758228,0.795743,0.831634,0.865438,0.896662,0.924792,0.949291,0.969609,0.985185,0.995457,0.999869,0.997879,0.988974,0.972677,0.948561,0.916266,0.875507,0.826096,0.767950,0.701112,0.625763,0.542235,0.451022,0.352797,0.248409,0.138898,0.025488,0.090412,0.207218,0.323186,0.436429,0.544947,0.646657,0.739435,0.821158,0.889756,0.943273,0.979921,0.998149,0.996706,0.974703,0.931678,0.867649,0.783161,0.679326,0.557846,0.421014,0.271710,0.113364,0.050100,0.214342,0.374710,0.526364,0.664422,0.784128,0.881027,0.951148,0.991197,0.998728,0.972313,0.911681,0.817819,0.693040,0.540992,0.366614,0.176028,0.023633,0.224459,0.418059,0.595908,0.749728,0.871903,0.955888,0.996609,0.990826,0.937421,0.837606,0.695026,0.515730,0.308005,0.082071,0.150363,0.376672,0.583990,0.759950,0.893467,0.975491,0.999706,0.963096,0.866347,0.714028,0.514534,0.279742,0.024416,0.234646,0.479684,0.693188,0.859186,0.964520,0.999988,0.961275,0.849560,0.671729,0.440149,0.171964,0.112052,0.389016,0.635769,0.830839,0.956392,1.000000,0.956030,0.826508,0.621327,0.357721,0.059016,0.247296,0.532030,0.767077,0.928261,0.997986,0.967336,0.837430,0.619777,0.335550,0.013751,0.311576,0.604883,0.833026,0.969125,0.995941,0.908314,0.714312,0.434853,0.101769,0.245611,0.564983,0.816220,0.966553,0.995065,0.895912,0.679728,0.372907,0.014721,0.347464,0.664345,0.891446,0.995568,0.959926,0.787244,0.500132,0.138562,0.245366,0.594759,0.856373,0.988891,0.969804,0.799641,0.502771,0.124417,0.275856,0.633347,0.888665,0.997845,0.940322,0.723269,0.381404,0.028055,0.434607,0.766458,0.963336,0.987793,0.832828,0.524248,0.117038,0.313975,0.687635,0.931764,0.997436,0.869318,0.569850,0.156011,0.290913,0.681036,0.933960,0.995691,0.850767,0.526767,0.089858,0.368130,0.748727,0.968077,0.975873,0.767576,0.386893,0.082596,0.535256,0.866970,0.999378,0.898988,0.586457,0.133352,0.353531,0.756929,0.977448,0.958585,0.702034,0.268991,0.233546,0.678670,0.951584,0.979647,0.752588,0.326978,0.186829,0.652697,0.944643,0.981389,0.749880,0.310794,0.216261,0.684673,0.961521,0.965828,0.693231,0.219350,0.320092,0.767522,0.989574,0.917547,0.569782,0.048444,0.489469,0.878912,0.997754,0.805982,0.360444,0.200633,0.699652,0.975716,0.937158,0.593190,0.053440,0.505738,0.898397,0.991125,0.749477,0.252181,0.333091,0.805333,0.999362,0.844457,0.391628,0.201503,0.724530,0.988650,0.895625,0.475821,0.119917,0.673151,0.977629,0.916863,0.510193,0.091355,0.659786,0.976102,0.915394,0.497718,0.116465,0.686436,0.985207,0.890693,0.437261,0.194673,0.749000,0.997449,0.834538,0.324215,0.323167,0.836294,0.996534,0.732360,0.153282,0.493534,0.927760,0.957690,0.566581,0.076263,0.686872,0.991479,0.850156,0.323078,0.352738,0.868466,0.984712,0.644052,0.001701,0.643100,0.985610,0.860420,0.323466,0.370438,0.886859,0.973003,0.583132,0.094762,0.727420,0.999407,0.771555,0.154548,0.542234,0.964491,0.893693,0.362338,0.357986,0.893530,0.962277,0.524316,0.192395,0.808296,0.992776,0.643123,0.055521,0.725542,1.000000,0.724810,0.047999,0.657020,0.996227,0.776113,0.116879,0.610243,0.990388,0.802679,0.151208,0.589399]

plt.figure('CTF')
plt.plot(freqs, CTF)

psf = np.real(fftpack.ifftshift(fftpack.ifft(np.array(CTF))))
plt.figure('PSF')
plt.plot(psf)
kernel_1d = cp.asarray(psf)#/cp.max(psf)

conv = Convolve(
    arg_shape=full_sino_ordered.shape,
    kernel=[cp.array([1]), kernel_1d, kernel_1d],
    center=[0, kernel_1d.size // 2, kernel_1d.size // 2],
    mode="constant",
    enable_warnings=True,
)

'''
plt.figure()
plt.imshow(proj_fft_abs2)
plt.show()

breakpoint()'''


#plt.show()

'''
sigmaa = 300
psf = np.exp(-np.linspace(-50,50)**2/sigmaa)
#psf = np.sin(np.linspace(-50,50)/sigmaa)/np.linspace(-50,50)

#plt.plot(psf.get())
#plt.show()




k1,k2,k3 = full_sino_ordered.shape
width, N_side, N_depth = k1,k2,k3
full_sino_ordered *= 0
full_sino_ordered[k1//2-k1//4:k1//2+k1//4 , k2//2-k2//4:k2//2+k2//4, k3//2-k3//4:k3//2+k3//4] = 1
plt.figure("phantom")
plt.imshow(full_sino_ordered.get()[k1//2 , :, :])

full_sino_ordered = conv(full_sino_ordered.reshape(-1))
plt.figure("data")
plt.imshow(full_sino_ordered.get().reshape(k1,k2,k3)[k1//2, :, :])
'''

'''
plt.figure()
plt.imshow(full_sino_ordered[23].get())

im = conv(full_sino_ordered.reshape(-1)).reshape(full_sino_ordered.shape)[23]
plt.figure()
plt.imshow(im.get())
plt.show()
'''

op_para_u = op_para_u # operator such that : y = Xray(f)*h, where * is the convolution
#-----------------
I_BP = op_para_u.adjoint(cp.array(full_sino_ordered.reshape(-1))).reshape(width, N_side, N_depth)
plt.figure('BP')
plt.imshow(I_BP.get()[width//2, N_side//2-512:N_side//2+512,:])



#fwd = op_para_u.apply(cp.array(I_BP.reshape(-1))).reshape(full_sino_ordered.shape)

'''
plt.figure('BP')
plt.imshow(I_BP[width//2,:,:])
plt.figure()
plt.imshow(I_BP[:,N_side//2,:])
plt.figure()
plt.imshow(I_BP[:,:,N_depth//2])
plt.show()

plt.figure('fwd')
plt.imshow(fwd[29,:,:])
plt.figure()
plt.imshow(fwd[:,N_side//2,:])
plt.figure()
plt.imshow(fwd[:,:,N_depth//2])
'''




#mask to cut full_sino_ordered where fwd is 0
#full_sino_ordered = cp.where(cp.array(fwd) == 0, 0, cp.array(full_sino_ordered))
#full_sino_ordered = cp.log(full_sino_ordered/cp.max(full_sino_ordered))


####uncomment for pinv

#I_BP = op_para_u.adjoint(cp.array(full_sino_ordered.reshape(-1))).reshape(width, N_side, N_depth)

plt.figure('BP')
plt.imshow(I_BP.get()[width//2, N_side//2-512:N_side//2+512,:])

fwd = op_para_u.apply(cp.array(I_BP.reshape(-1))).reshape(full_sino_ordered.shape)
plt.figure('fwd of bp')
plt.imshow(fwd.get()[4,:,:])


stop_crit = pxst.MaxIter(5) #500
recon_pinv = op_para_u.pinv(cp.array(full_sino_ordered.reshape(-1)), damp=100, kwargs_fit=dict(stop_crit=stop_crit)).reshape(width, N_side, N_depth)
fwd = op_para_u.apply(cp.array(recon_pinv.reshape(-1))).reshape(full_sino_ordered.shape)
plt.figure('pinv')
plt.imshow(recon_pinv.get()[width//2,N_side//2-512:N_side//2+512,3:-3])





'''
sigma_torch = torch.tensor(2).to(device).view(1, 1, 1, 1)

arg_shape = (width, N_side, N_depth)
def denoiser_grad(arr):
    with torch.no_grad():
        arr_reshaped = arr.reshape(arg_shape)
        stack_denoised = torch.clone(arr_reshaped)
        for w in range(width):
            arr = arr_reshaped[w]
            im_denoised, _, _ = utils.accelerated_gd_batch(arr.reshape(1, *(1,N_side, N_depth)), model, sigma=sigma_torch, ada_restart=True, tol=1e-4)
            stack_denoised[w] = im_denoised
    return (stack_denoised).ravel()
    return (arr - stack_denoised).ravel()

nn_denoiser = from_torch(
    apply=None,
    grad=denoiser_grad,
    shape=(1, width*N_side*N_depth),
    cls=pxa.ProxDiffFunc,
    dtype="float32",
    enable_warnings=True,
    name='WCRR',
)

#nn_denoiser = blocks.hstack(width * [nn_denoiser])
print(nn_denoiser)

nn_denoiser.diff_lipschitz = 2.
sigma = 1
loss = (1/ (2 * sigma**2)) * SquaredL2Norm(dim=cp.array(full_sino_ordered).size).asloss(cp.array(full_sino_ordered.ravel())) * op_para_u
solver = pgd.PGD(f=loss+ 100* nn_denoiser)

#solver.fit(x0=recon_pinv.reshape(-1), stop_crit=pxst.MaxIter(5))

alpha = 0.0001
dnn_lambda = 0.03
x = cp.array(0*recon_pinv.reshape(-1))
for k in range(10):
    print(k)
    x_next = x - alpha*( op_para_u.adjoint(op_para_u.apply(x)) - op_para_u.adjoint(cp.array(full_sino_ordered.reshape(-1))))
    x_next = x_next - dnn_lambda* (x - nn_denoiser.grad(x))
    x = x_next
    x_pnp = x_next.get().reshape(arg_shape) #solver.solution().get().reshape(arg_shape)
    if k%3==0:
        plt.figure('pnp ' + str(k))
        plt.imshow(x_pnp[width//2,N_side//2-512:N_side//2+512,:])
plt.show()
 
breakpoint()
'''





#plt.show()
plt.figure('fwd hor')
plt.imshow(fwd.get()[4,:,:])
plt.figure('sino hor')
plt.imshow(full_sino_ordered.get()[4,:,:])
plt.figure('fwd vert')
plt.imshow(fwd.get()[:,100,:])
plt.figure('sino vert')
plt.imshow(full_sino_ordered.get()[:,100,:])
plt.figure('fwd sag')
plt.imshow(fwd.get()[:,:,100])
plt.figure('sino sag')
plt.imshow(full_sino_ordered.get()[:,:,100])


#plt.show()

#breakpoint()



'''
stop_crit = pxst.MaxIter(1000)
recon_pinv = op_para_u.pinv(cp.array(full_sino_ordered.reshape(-1)), damp=500, kwargs_fit=dict(stop_crit=stop_crit)).reshape(width, N_side, N_depth)

cp.save('recon_tv.npy', recon_pinv[width//4,:,:])

breakpoint()

'''



'''
plt.figure('full_sino_ordered')
plt.imshow(full_sino_ordered[29,:,:])
plt.figure()
plt.imshow(full_sino_ordered[:,N_side//2,:])
plt.figure()
plt.imshow(full_sino_ordered[:,:,N_depth//2])
plt.show()

#convert cp arrays:
plt.figure('BP')
plt.imshow(I_BP[width//2,:,:])
plt.figure()
plt.imshow(I_BP[:,N_side//2,:])
plt.figure()
plt.imshow(I_BP[:,:,N_depth//2])
plt.show()

fwd = op_para_u.apply(cp.array(I_BP.reshape(-1))).reshape(full_sino_ordered.shape)
plt.figure('fwd')
plt.imshow(fwd[29,:,:])
plt.figure()
plt.imshow(fwd[:,N_side//2,:])
plt.figure()
plt.imshow(fwd[:,:,N_depth//2])
plt.show()


stop_crit = pxst.MaxIter(10)
recon_pinv = op_para_u.pinv(cp.array(full_sino_ordered.reshape(-1)), damp=500, kwargs_fit=dict(stop_crit=stop_crit)).reshape(width, N_side, N_depth)

plt.figure('recon_pinv')
plt.imshow(recon_pinv[width//2,:,:])
plt.show()

breakpoint()

'''

 

######## HERE for pinv and adjoint . Works but very noisy.
######## Many iterations lead to a very noisy image (fit of the noise in the data)


#Try with TV prior

# TV prior
x0=cp.array(cp.zeros((width, N_side, N_depth))).ravel()

#grad = Gradient(arg_shape=(width, N_side, N_depth), sample = [pitch, pitch, pitch], mode = 'reflect', gpu=True)
lambda_ = 2
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
        finite_diff_op.lipschitz= 400 # finite_diff_op.estimate_lipschitz()
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
        
        ls.diff_lipschitz = 400 # ls.estimate_diff_lipschitz()
        print("-------------------", ls.diff_lipschitz)
        slv = CV(f=ls, h=tau * self._norm, K=self._finite_diff_op, verbosity = 1)
        slv.fit(x0=arr.copy(), stop_crit = pxst.RelError(eps=1e-90, var="x", f=None, norm=2, satisfy_all=True) | pxst.MaxIter(20))
        return slv.solution().reshape(arr.shape)

# Loss
v_shape = (width, N_side, N_depth)
tv_prior = TVFunc(arg_shape=v_shape)
sigma = 1
loss = (1/ (2 * sigma**2)) * SquaredL2Norm(dim=cp.array(full_sino_ordered).size).asloss(cp.array(full_sino_ordered.ravel())) * op_para_u
# Smooth part of the posterior
smooth_posterior = loss 
smooth_posterior.diff_lipschitz = 1e4

#l21 = L21Norm(arg_shape=(3, *x0.shape), l2_axis=(0, 1,2))  # Total variation prior


# Define the solver
solver = PGD(f=smooth_posterior, g = lambda_ * tv_prior, show_progress=True, verbosity=1)
#solver = PD3O(f=smooth_posterior, g = posL1, h=lambda_ * l21, K=grad, show_progress=True, verbosity=1)

# Call fit to trigger the solver
stop_crit = pxst.RelError(eps=1e-14, var="x", f=smooth_posterior, norm=2, satisfy_all=True) | pxst.MaxIter(10)

solver.fit(x0=recon_pinv.reshape(-1), stop_crit=stop_crit, acceleration=True)

recon_tv = solver.solution().squeeze()

recon_tv = recon_tv.reshape(width, N_side, N_depth)

plt.figure('TV')
plt.imshow(recon_tv.get()[width//2,N_side//2-512:N_side//2+512,:])
plt.show()
#cp.save('recon_tv.npy', recon_tv[width//4,:,:])

breakpoint()


plt.figure('BP')
plt.imshow(I_BP[width//2,:,:])

plt.figure('recon_tv')
plt.imshow(recon_tv[width//2,:,:])

plt.figure()
plt.imshow(recon_tv[width//4,:,:])

plt.figure()
plt.imshow(recon_tv[(3*width)//4,:,:])

plt.show()


breakpoint()

plt.figure('recon_tv')
plt.imshow(recon_tv[N_side//2,:,:])
plt.show()

breakpoint()