import time
import torch
import random
import numpy as np
import torch.nn as nn
from training.utils import utils
from training.utils.utilities import batch_PSNR
from matplotlib import pyplot as plt

device = "cuda:3"
fname = 'CRR-CNN_32_channels_3_kernelsize_15_fwsteps_3_bwsteps'
model = utils.load_model(fname, device=device)
model.eval()
model.to(device)
print(" **** Updating the Lipschitz constant **** ")
model.conv_layer.L = 83.1117
#sn_pm = model.conv_layer.spectral_norm(mode="power_method", n_steps=500)
#print("Spectral norm: ", sn_pm)
L_R = model.nonlinearity.slopes.max()*model.get_mu()