import torch
from torch.distributions import Distribution

class Normal(Distribution):
    def __init__(self, loc, scale):
        self.loc = loc
        self.scale = scale

    def transform(self, z):
        return (z - self.loc) / self.scale
    
    def d_transform_d_z(self):
        return 1 / self.scale

    def sample(self):
        return torch.normal(self.loc, self.scale).detach()

    def rsample(self):
        x = self.sample()

        transform = self.transform(x)

        surrogate_x = - transform / self.d_transform_d_z().detach()

        # Replace gradients of x with gradients of surrogate_x, but keep the value.
        return x + (surrogate_x - surrogate_x.detach())
