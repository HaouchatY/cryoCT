import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.parametrize as P

class MultiConv3d(nn.Module):
    def __init__(self, size_kernels, num_channels):

        super().__init__()
        # parameters and options
        self.size_kernels =  size_kernels
        self.num_channels = num_channels
        
        # list of convolutionnal layers
        self.conv_layers = nn.ModuleList()
        bias = False
        
        for j in range(len(self.num_channels) - 1):
            self.conv_layers.append(nn.Conv3d(self.num_channels[j], self.num_channels[j+1], self.size_kernels[j], \
                                              padding=self.size_kernels[j]//2, bias=bias))
        P.register_parametrization(self.conv_layers[0], "weight", ZeroMean())

    def forward(self, x):

        for conv in self.conv_layers:
            x = F.conv3d(x, conv.weight, padding=conv.padding)
 
        return x
    
    def transpose(self, x):

        for conv in reversed(self.conv_layers):
            x = F.conv_transpose3d(x, conv.weight, padding=conv.padding)

        return x

class ZeroMean(nn.Module):
    def forward(self, x):
        return x - torch.mean(x, dim=(1,2,3,4), keepdim=True)