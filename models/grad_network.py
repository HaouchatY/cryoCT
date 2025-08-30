import torch
import torch.nn as nn
from models.multi_conv import MultiConv3d
from models.linearspline import LinearSpline
import torch.nn.functional as F

class WCvxConvNet(nn.Module):
    def __init__(self, multi_convolution, training_options, spline_scaling):
        super(WCvxConvNet, self).__init__()

        # Convolution Hyperparameters
        self.conv = MultiConv3d(multi_convolution['size_kernels'], multi_convolution['num_channels'])
        self.nb_filters = multi_convolution['num_channels'][-1]
        self.filter_size = sum(multi_convolution['size_kernels']) - len(multi_convolution['size_kernels']) + 1        
        self.dirac = torch.zeros(1, 1, 2*self.filter_size-1, 2*self.filter_size-1, 2*self.filter_size-1)
        self.dirac[0, 0, self.filter_size-1, self.filter_size-1, self.filter_size-1] = 1.0

        # Nonlinearity Parameters
        self.mu = nn.Parameter(torch.tensor(4.0, dtype=torch.float32))
        self.beta = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))

        # DEQ Hyperparameters
        self.f_max_iter = training_options['f_max_iter']
        self.b_max_iter = training_options['b_max_iter']

        # Spline Parameters
        self.spline_scaling = LinearSpline(**spline_scaling)
    
    def activation(self, x):
        mu = self.mu.exp()

        x = torch.sign(x) * torch.relu(torch.minimum(mu * torch.abs(x), self.beta -  torch.abs(x))) \
            + torch.relu(torch.abs(x) - self.beta)
        
        return x

    def grad(self, x, sigma):
        grad = self.conv(x)
        scaling = self.spline_scaling(sigma)
        grad = grad * scaling
        grad = self.activation(grad)
        grad = grad / scaling
        grad = self.conv.transpose(grad)
        return grad

    def grad_split(self, x, sigma, split=4):

        if not hasattr(self, 'impulses'):
            dirac = torch.zeros(1, 1, self.filter_size, self.filter_size, self.filter_size, device=x.device)
            dirac[0, 0, self.filter_size//2, self.filter_size//2, self.filter_size//2] = 1.0
            self.impulses = self.conv(dirac).permute(1, 0, 2, 3, 4).flip(2, 3, 4)
        import matplotlib.pyplot as plt
        # plot the filters
        fig, axs = plt.subplots(1, split, figsize=(20, 5))
        for i in range(split):
            axs[i].imshow(self.impulses[i].detach().cpu().numpy()[0, 0, self.filter_size//2])
            axs[i].axis('off')
        plt.show()
        return 0


        grad = torch.zeros_like(x)
        split_size = self.nb_filters // split
        #sigma_ = sigma.repeat(1, self.nb_filters, 1, 1, 1)
        scaling = self.spline_scaling(sigma)

        #time profiling 
        
        for k in range(split):

            temp_grad = F.conv3d(x, self.impulses[k*split_size:(k+1)*split_size], padding=self.filter_size//2)
            temp_grad = temp_grad * scaling[:, k*split_size:(k+1)*split_size]
            temp_grad = self.activation(temp_grad)
            temp_grad = temp_grad / scaling[:, k*split_size:(k+1)*split_size]
            #temp_grad = F.conv_transpose3d(temp_grad, self.impulses[k*split_size:(k+1)*split_size], padding=self.filter_size//2)
            temp_grad = F.conv3d(temp_grad, self.impulses[k*split_size:(k+1)*split_size].permute(1, 0, 2, 3, 4).flip(2, 3, 4),\
                                  padding=self.filter_size//2)
            grad += temp_grad

        return grad
    
    def grad_lipschitz(self):
        impulse = self.conv(self.dirac)
        impulse = torch.maximum(self.mu.exp(), torch.ones_like(self.mu))*impulse
        impulse = self.conv.transpose(impulse)
        frequencies = torch.fft.fftn(impulse, s=[256, 256, 256], dim=(2,3,4))
        return frequencies.abs().max()
    
    def _apply(self, fn):
        self.dirac = fn(self.dirac)
        return super()._apply(fn)