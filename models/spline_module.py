import torch
import torch.nn as nn
from abc import ABC


class LinearSpline(ABC, nn.Module):
    """
    Class for LinearSpline activation functions

    Args:
        num_knots (int): number of knots of the spline
        num_activations (int) : number of activation functions
        x_min (float): position of left-most knot
        x_max (float): position of right-most knot
        slope_min (float or None): minimum slope of the activation
        slope_max (float or None): maximum slope of the activation
        antisymmetric (bool): Constrain the potential to be symmetric <=> activation antisymmetric
    """

    def __init__(self, num_activations, num_knots, x_min, x_max, init,
                 slope_max=None, slope_min=None, antisymmetric=False, clamp=True, **kwargs):

        super().__init__()

        self.num_knots = int(num_knots)
        self.num_activations = int(num_activations)
        self.init = init
        self.x_min = torch.tensor([x_min])
        self.x_max = torch.tensor([x_max])

        self.step_size = (self.x_max - self.x_min) / (self.num_knots - 1)
        self.clamp = clamp
        
        # parameters
        coefficients = self.initialize_coeffs()  # spline coefficients
        self.coefficients = nn.Parameter(coefficients)

        self.init_zero_knot_indexes()
            

    def init_zero_knot_indexes(self):
        """ Initialize indexes of zero knots of each activation.
        """
        # self.zero_knot_indexes[i] gives index of knot 0 for filter/neuron_i.
        # size: (num_activations,)
        activation_arange = torch.arange(0, self.num_activations)
        self.zero_knot_indexes = (activation_arange * self.num_knots)

    def initialize_coeffs(self):
        """The coefficients are initialized with the value of the activation
        # at each knot (c[k] = f[k], since B1 splines are interpolators)."""
        init = self.init
        grid_tensor = torch.linspace(self.x_min.item(), self.x_max.item(), self.num_knots).expand((self.num_activations, self.num_knots))

        coefficients = torch.ones_like(grid_tensor)*init
        return coefficients

    def forward(self, x):
        """
        Args:
            input (torch.Tensor):
                2D or 4D, depending on weather the layer is
                convolutional ('conv') or fully-connected ('fc')

        Returns:
            output (torch.Tensor)
        """

        in_shape = x.shape
        in_channels = in_shape[1]

        if in_channels % self.num_activations != 0:
            raise ValueError('Number of input channels must be divisible by number of activations.')

        x = x.view(x.shape[0], self.num_activations, in_channels // self.num_activations, *x.shape[2:])

        x = LinearSpline_Func.apply(x, self.coefficients, self.x_min, self.x_max, self.num_knots, self.zero_knot_indexes)

        x = x.view(in_shape)

        return x
    
    def _apply(self, fn):
        self.x_min = fn(self.x_min)
        self.x_max = fn(self.x_max)
        self.step_size = fn(self.step_size)
        self.zero_knot_indexes = fn(self.zero_knot_indexes)
        return super()._apply(fn)

    def extra_repr(self):
        """ repr for print(model) """

        s = ('num_activations={num_activations}, '
             'init={init}, num_knots={num_knots}, range=[{x_min[0]:.3f}, {x_max[0]:.3f}]. '
             )

        return s.format(**self.__dict__)
    
class LinearSpline_Func(torch.autograd.Function):
    """
    Autograd function to only backpropagate through the B-splines that were
    used to calculate output = activation(input), for each element of the
    input.
    """
    @staticmethod
    def forward(ctx, x, coefficients, x_min, x_max, num_knots, zero_knot_indexes):

        # The value of the spline at any x is a combination 
        # of at most two coefficients
        step_size = (x_max - x_min) / (num_knots - 1)
        x_clamped = x.clamp(min=x_min.item(), max=x_max.item() - step_size.item())


        floored_x = torch.floor((x_clamped - x_min) / step_size)  #left coefficient

        fracs = (x - x_min) / step_size - floored_x  # distance to left coefficient

        # This gives the indexes (in coefficients_vect) of the left
        # coefficients
        indexes = (zero_knot_indexes.view(1, -1, 1, 1, 1, 1) + floored_x).long()

        coefficients_vect = coefficients.view(-1)

        # Only two B-spline basis functions are required to compute the output
        # (through linear interpolation) for each input in the B-spline range.
        activation_output = coefficients_vect[indexes + 1] * fracs + \
            coefficients_vect[indexes] * (1 - fracs)

        ctx.save_for_backward(fracs, coefficients, indexes, step_size)
        # ctx.results = (fracs, coefficients_vect, indexes, grid)
        return activation_output

    @staticmethod
    def backward(ctx, grad_out):

        fracs, coefficients, indexes, step_size = ctx.saved_tensors

        coefficients_vect = coefficients.view(-1)

        grad_x = (coefficients_vect[indexes + 1] -
                  coefficients_vect[indexes]) / step_size * grad_out

        # Next, add the gradients with respect to each coefficient, such that,
        # for each data point, only the gradients wrt to the two closest
        # coefficients are added (since only these can be nonzero).
        grad_coefficients_vect = torch.zeros_like(coefficients_vect, dtype=coefficients_vect.dtype)
        # right coefficients gradients
   

        grad_coefficients_vect.scatter_add_(0,
                                            indexes.view(-1) + 1,
                                            (fracs * grad_out).view(-1))
        # left coefficients gradients
        grad_coefficients_vect.scatter_add_(0, indexes.view(-1),
                                            ((1 - fracs) * grad_out).view(-1))

        grad_coefficients = grad_coefficients_vect.view(coefficients.shape)

        return grad_x, grad_coefficients, None, None, None, None