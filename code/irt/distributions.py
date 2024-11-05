import torch
from torch.distributions import Distribution


# Define a custom Normal distribution class that inherits from PyTorch's Distribution class
class Normal(Distribution):
    # Indicates that the distribution supports reparameterized sampling
    has_rsample = True

    def __init__(self, loc: torch.Tensor, scale: torch.Tensor, generator: torch.Generator = None) -> None:
        """
        Initializes the Normal distribution with a given mean (loc) and standard deviation (scale).

        Args:
            loc (Tensor): Mean of the normal distribution. This defines the central tendency of the distribution.
            scale (Tensor): Standard deviation of the normal distribution. This defines the spread or width of the distribution.
            generator (torch.Generator, optional): A random number generator for reproducible sampling.
        """
        self.loc = loc  # Mean of the distribution
        self.scale = scale  # Standard deviation of the distribution
        self.generator = generator  # Optional random number generator for reproducibility
        super(Distribution).__init__()  # Initialize the base Distribution class

    def transform(self, z: torch.Tensor) -> torch.Tensor:
        """
        Transforms the input tensor `z` to the standard normal form using the distribution's mean and scale.

        Args:
            z (Tensor): Input tensor to be transformed.

        Returns:
            Tensor: The transformed tensor, which is normalized to have mean 0 and standard deviation 1.
        """
        return (z - self.loc) / self.scale

    def d_transform_d_z(self) -> torch.Tensor:
        """
        Computes the derivative of the transform function with respect to the input tensor `z`.

        Returns:
            Tensor: The derivative, which is the reciprocal of the scale. This is used for reparameterization.
        """
        return 1 / self.scale

    def sample(self) -> torch.Tensor:
        """
        Generates a sample from the Normal distribution using PyTorch's `torch.normal` function.

        Returns:
            Tensor: A tensor containing a sample from the distribution. The `detach()` method is used to prevent
                    gradients from being tracked during sampling.
        """
        return torch.normal(self.loc, self.scale, generator=self.generator).detach()

    def rsample(self) -> torch.Tensor:
        """
        Generates a reparameterized sample from the Normal distribution, which is useful for gradient-based optimization.

        The `rsample` method generates a sample `x`, applies a transformation, and creates a surrogate sample
        that allows gradients to flow through the sampling process.

        Returns:
            Tensor: A reparameterized sample tensor, which allows gradient backpropagation.
        """
        x = self.sample()  # Sample from the distribution
        
        transform = self.transform(x)  # Transform the sample to standard normal form
        surrogate_x = -transform / self.d_transform_d_z().detach()  # Compute the surrogate for backpropagation
        # Return the sample adjusted to allow gradient flow
        return x + (surrogate_x - surrogate_x.detach())
