import torch
import torch.nn as nn
import torch.nn.utils.parametrize as P
from models.multi_conv import MultiConv3d
from models.spline_module import LinearSpline
from models.nonlinearity import ConvexGrad, WeaklyConvexGrad

class WCvxConvNet(nn.Module):
    def __init__(self, param_multi_conv, param_spline_scaling):
        
        super().__init__()

        self.num_channels = param_multi_conv["num_channels"][-1]
        # 1 - multi-convolutionnal layers
        print("Multi convolutionnal layer: ", param_multi_conv)
        self.conv_layer = MultiConv3d(**param_multi_conv)

        # 2 - activation functions (gradient of the potential)
        self.nonlinearity = WeaklyConvexGrad(self.num_channels)
        
        # 3 - mu parameter (controls the magnitude of the convex part of the potential / increasing part of spline)
        self.mu_ = nn.Parameter(torch.tensor(4.0, dtype=torch.float32))
        
        # 4 - scaling parameter to add some flexibility to the activation accross channels and noise levels
        self.spline_scaling = LinearSpline(**param_spline_scaling)
        
        self.num_params = sum(p.numel() for p in self.parameters())

        # to cache the scaling and mu
        self.scaling = None

        # cached values
        self.cached_wx = None

    @property
    def device(self):
        return(self.conv_layer.conv_layers[0].weight.device)
        
    def get_scaling(self, sigma=None):
        if self.scaling is None:
            eps = 1e-5
            return(torch.exp(self.spline_scaling(torch.tile(sigma, (1, self.num_channels, 1, 1, 1)))) / (sigma + eps))
        else:
            return(self.scaling)
        
    def cache_scaling(self, sigma=None):
        self.scaling = self.get_scaling(sigma)

    def clear_scaling(self):
        self.scaling = None


    def get_mu(self):
        return(self.mu_.exp())

    def activation(self, x, sigma=None, skip_scaling=False):
        # get scaling, which depends on sigma and on the channel
        if not skip_scaling:
            scaling = self.get_scaling(sigma)
        else:
            scaling = 1
        x = x * scaling
        # apply activation

        y = self.get_mu() * self.nonlinearity(x, self.get_mu()) 
        # scale back
        y = y / scaling

        return(y)

    def grad_activation(self, x, sigma=None):
        scaling = self.get_scaling(sigma)
        x = x * scaling
        return self.get_mu() * self.nonlinearity.derivative(x, self.get_mu())


    def integrate_activation(self, x, sigma=None, skip_scaling=False):

        if not skip_scaling:
            scaling = self.get_scaling(sigma)
        else:
            scaling = 1

        x = x * scaling

        y = self.get_mu() * self.nonlinearity.integrate(x)

        y = y / scaling / scaling

        return(y)
    
    '''def grad(self, x, sigma=None, cache_wx=False, gpusize_fit=200):
        """ Gradient of the loss at location x. Update conv.L before if needed."""
        # Get the shape of the input tensor
        s = x.shape
        # Calculate the number of subvolumes in dimensions 3 and 4
        num_subvolumes_3 = s[3] // gpusize_fit
        num_subvolumes_4 = s[4] // gpusize_fit
        # Calculate the overlap size
        overlap = 10
        # Initialize an empty tensor to store the subvolume gradients
        grad_subvolumes = torch.zeros_like(x)
        
        # Iterate over each subvolume
        for i in range(num_subvolumes_3):
            for j in range(num_subvolumes_4):
                # Calculate the start and end indices for the subvolume
                start_3 = i * gpusize_fit
                end_3 = start_3 + gpusize_fit
                start_4 = j * gpusize_fit
                end_4 = start_4 + gpusize_fit
                
                # Extract the subvolume
                subvolume = x[:, :, :, start_3:end_3, start_4:end_4]
                
                # Compute the gradient for the subvolume
                grad_subvolume = self._compute_subvolume_gradient(subvolume, sigma, cache_wx)
                
                # Update the corresponding region in the grad_subvolumes tensor
                grad_subvolumes[:, :, :, start_3:end_3, start_4:end_4] = grad_subvolume
        
        # Combine the subvolume gradients with overlap
        grad_combined = self._combine_subvolume_gradients(grad_subvolumes, overlap)
        
        return grad_combined
    
    def _compute_subvolume_gradient(self, subvolume, sigma=None, cache_wx=False):
        """Compute the gradient for a subvolume"""
        # First multi convolution layer
        y = self.conv_layer(subvolume)
        
        if cache_wx:
            self.cached_wx = y
        
        # Activation
        y = self.activation(y, sigma=sigma)
        
        # Transpose convolution
        y = self.conv_layer.transpose(y)
        
        return y
    
    def _combine_subvolume_gradients(self, grad_subvolumes, overlap):
        """Combine the subvolume gradients with overlap"""
        # Get the shape of the input tensor
        s = grad_subvolumes.shape
        # Calculate the start and end indices for the overlapping region
        start_3 = overlap
        end_3 = s[3] - overlap
        start_4 = overlap
        end_4 = s[4] - overlap
        # Initialize an empty tensor to store the combined gradients
        grad_combined = torch.zeros_like(grad_subvolumes)
        
        # Copy the gradients from the subvolumes to the combined tensor
        grad_combined[:, :, :, :start_3, :] = grad_subvolumes[:, :, :, :start_3, :]
        grad_combined[:, :, :, end_3:, :] = grad_subvolumes[:, :, :, end_3:, :]
        grad_combined[:, :, :, start_3:end_3, :start_4] = grad_subvolumes[:, :, :, start_3:end_3, :start_4]
        grad_combined[:, :, :, start_3:end_3, end_4:] = grad_subvolumes[:, :, :, start_3:end_3, end_4:]
        
        return grad_combined'''
    
    def grad(self, x, sigma=None, cache_wx=False):

        """ Gradient of the loss at location x. Update conv.L before if needed."""
        # first multi convolution layer
        y = self.conv_layer(x)

        if cache_wx:
            self.cached_wx = y
        
        # activation
        y = self.activation(y, sigma=sigma)

        y =  self.conv_layer.transpose(y)
 
        return(y)

    def grad_denoising(self, x, x_noisy, sigma=None, cache_wx=False, lmbd=1):

        return(1/(1 + lmbd*self.get_mu()) * ((x - x_noisy) + lmbd*self.grad(x, sigma=sigma, cache_wx=cache_wx)))

    def hvp(self, x, v, sigma=None):

        """ Hessian of R vector product """
        # first multi convolution layer on x and v
        y_x = self.conv_layer(x)
        y_v = self.conv_layer(v)

        # derivative activation at y_v
        y_x_1 = self.grad_activation(y_x, sigma=sigma)

        y = y_x_1 * y_v

        y =  self.conv_layer.transpose(y)

        return(y)

    def hvp_denoising(self, x, v, sigma=None):

        return(1/(1 + self.get_mu()) * (v + self.hvp(x, v, sigma=sigma)))


    def cost(self, x, sigma, use_cached_wx=False):
        s = x.shape
        # first multi convolution layer
        
        if use_cached_wx:
            y = self.cached_wx
        else:
            y = self.conv_layer(x)
        #print(y.shape)
        # activation
        y = self.integrate_activation(y, sigma)
        #print(y.shape)

        return(torch.sum(y, dim=tuple(range(1, len(s)))))