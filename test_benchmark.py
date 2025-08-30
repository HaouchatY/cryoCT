import numpy as np
import matplotlib.pyplot as plt
import cupy as cp
import torch
import random
import numpy as np
import torch.nn as nn
from training.utils import utils
from training.utils.utilities import batch_PSNR
from models.grad_network import WCvxConvNet

# use cmap = 'gray' always with plt
plt.rc('image', cmap='gray')

torch.set_grad_enabled(False)

model_name = 'WCRR-CNN_f_30_b_5_multispline'
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
PATH = f'cryo-ct/trained_models/{model_name}/checkpoints/checkpoint.pth'
infos = torch.load(PATH)
config = infos['config']
model = WCvxConvNet(config['multi_convolution'], config['training_options'], config['spline_scaling'])
model.load_state_dict(infos['state_dict'], strict=False)
model = model.to(device)

L_R = model.grad_lipschitz()

print(f'L_R = {L_R}')


v_test = np.ones((500,500,500))
v_shape = v_test.shape
sigma_ = 30.


def grad_subvolume(model, x, sigma=None, cache_wx=False, N_subvolumes=1):
            
    """ Gradient of the loss at location x. Update conv.L before if needed."""
    # first multi convolution layer
    grad = torch.zeros_like(x)
    with torch.no_grad():
        for i in range(N_subvolumes):
            # Dimensions are (batch, channels, depth, height, width)
            x_sub = x[:, :, :, i * x.shape[3] // N_subvolumes : (i + 1) * x.shape[3] // N_subvolumes, :]
            grad_sub = model.grad_split(x_sub, sigma=sigma, split=1)
            grad[:, :, :, i * x.shape[3] // N_subvolumes : (i + 1) * x.shape[3] // N_subvolumes, :] = grad_sub

    return(grad)


sigma = torch.tensor(sigma_).repeat(1,1,1,1,1).to(device)
v_test = torch.rand(1, 1, 300, 300, 300).to(device)
res1 = model.grad_split(v_test, sigma=sigma, split = 8)

print('----------------------------------')
res2 = model.grad_split(v_test, sigma=sigma, split = 8)
print('----------------------------------')

res2 = model.grad_split(v_test, sigma=sigma, split = 8)
print('----------------------------------')

res2 = model.grad_split(v_test, sigma=sigma, split = 8)
print('----------------------------------')

res2 = model.grad_split(v_test, sigma=sigma, split = 8)
print('----------------------------------')

res2 = model.grad_split(v_test, sigma=sigma, split = 8)
print('----------------------------------')

res2 = model.grad_split(v_test, sigma=sigma, split = 8)
print('----------------------------------')

res2 = model.grad_split(v_test, sigma=sigma, split = 8)
print('----------------------------------')

res2 = model.grad_split(v_test, sigma=sigma, split = 8)
print('----------------------------------')

res2 = model.grad_split(v_test, sigma=sigma, split = 8)

#res2 = model.grad_split(v_test, sigma=sigma, split = 8)
print(res1.shape, res2.shape)
print(res1.dtype, res2.dtype)
print(res1.device, res2.device)
import time
start = time.time()
res3 = res1.cpu().numpy()
res4 = res2.cpu().numpy()
print('subtraction time:', time.time() - start)