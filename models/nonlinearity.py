import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvexGrad(nn.Module):
    def __init__(self, num_activations):
        super().__init__()
        self.slopes = nn.Parameter(0.2*torch.ones(1, num_activations, 1, 1, 1))

    def forward(self, x, weak_limit):
        return torch.clip(self.slopes, 0., 1.)*torch.clip(x, -0.1, 0.1)

    def integrate(self, x):
        return(F.relu(self.slopes)/2 * (torch.relu(x + 0.1)**2 - torch.relu(x - 0.1)**2) - 0.1*x - (self.slopes*0.1)**2)
    
    def derivative(self, x, weak_limit):
        return torch.where(torch.abs(x) <= 0.1, F.relu(self.slopes), 0)
    

    def slope_max(self):
        return torch.max(torch.clip(self.slopes, 0., 1.), dim=1)[0]
    
    def extra_repr(self):
        return f"""nb_slopes={self.slopes.shape[1]}"""
    
class WeaklyConvexGrad(nn.Module):
    def __init__(self, num_activations):
        super().__init__()
        self.mu = nn.Parameter(0.2*torch.ones(1, num_activations, 1, 1, 1))
        self.gamma = nn.Parameter(0.01*torch.ones(1, num_activations, 1, 1, 1))
        self.beta = nn.Parameter(0.02*torch.ones(1, num_activations, 1, 1, 1))

    def forward(self, x, weak_limit):
        x_sg = torch.sign(x)
        x_abs = torch.abs(x)
        mu, beta = torch.clip(self.mu, 0., 1.), F.relu(self.beta)
        gamma = torch.clip(self.gamma, torch.tensor(0., device=x.device), 1./weak_limit)
        return x_sg * torch.relu(torch.minimum(mu * x_abs, beta - gamma * x_abs))

    def integrate(self, x):
        pass

    def derivative(self, x, weak_limit):
        mu, beta = torch.clip(self.mu, 0., 1.), F.relu(self.beta)
        gamma = torch.clip(self.gamma, torch.tensor(0., device=x.device), 1./weak_limit)
        return torch.where(torch.abs(x) <= beta/(mu+gamma), mu, torch.where(torch.abs(x) >= beta/gamma, 0., -gamma))
    
    def slope_max(self, weak_limit):
        mu, beta = torch.clip(self.mu, 0., 1.), F.relu(self.beta)
        gamma = torch.clip(self.gamma, torch.tensor(0., device=mu.device), 1./weak_limit)
        return torch.max(torch.maximum(mu, gamma), dim=1)[0]