import math
from numbers import Number, Real
from typing import Optional, List

import torch
import torch.nn.functional as F
from torch.distributions import constraints, Distribution
from torch.distributions.exp_family import ExponentialFamily
from torch.distributions.utils import broadcast_all
from torch.autograd.functional import jacobian
from torch.distributions import Distribution, Bernoulli, Binomial
from torch.distributions import ContinuousBernoulli, Geometric, NegativeBinomial, RelaxedBernoulli


class Normal(ExponentialFamily):
    """
    Represents the Normal (Gaussian) distribution with specified mean (loc) and standard deviation (scale).
    Inherits from PyTorch's ExponentialFamily distribution class.
    """
    
    has_rsample = True
    arg_constraints = {"loc": constraints.real, "scale": constraints.positive}
    support = constraints.real

    def __init__(
        self,
        loc: torch.Tensor,
        scale: torch.Tensor,
        validate_args: Optional[bool] = None,
    ) -> None:
        """
        Initializes the Normal distribution.

        Args:
            loc (torch.Tensor): Mean (location) parameter of the distribution.
            scale (torch.Tensor): Standard deviation (scale) parameter of the distribution.
            validate_args (Optional[bool]): If True, checks the distribution parameters for validity.
        """
        self.loc, self.scale = broadcast_all(loc, scale)
        # Determine batch shape based on the type of `loc` and `scale`.
        batch_shape = torch.Size() if isinstance(loc, Number) and isinstance(scale, Number) else self.loc.size()
        super().__init__(batch_shape, validate_args=validate_args)

    @property
    def mean(self) -> torch.Tensor:
        """
        Returns the mean of the distribution.
        
        Returns:
            torch.Tensor: The mean (location) parameter `loc`.
        """
        return self.loc

    @property
    def mode(self) -> torch.Tensor:
        """
        Returns the mode of the distribution.
        
        Returns:
            torch.Tensor: The mode (equal to `loc` in a Normal distribution).
        """
        return self.loc

    @property
    def stddev(self) -> torch.Tensor:
        """
        Returns the standard deviation of the distribution.
        
        Returns:
            torch.Tensor: The standard deviation (scale) parameter `scale`.
        """
        return self.scale

    @property
    def variance(self) -> torch.Tensor:
        """
        Returns the variance of the distribution.
        
        Returns:
            torch.Tensor: The variance, computed as `scale ** 2`.
        """
        return self.stddev.pow(2)

    def entropy(self) -> torch.Tensor:
        """
        Computes the entropy of the distribution.
        
        Returns:
            torch.Tensor: The entropy of the Normal distribution, which is a measure of uncertainty.
        """
        return 0.5 + 0.5 * math.log(2 * math.pi) + torch.log(self.scale)

    def cdf(self, value: torch.Tensor) -> torch.Tensor:
        """
        Computes the cumulative distribution function (CDF) of the distribution at a given value.

        Args:
            value (torch.Tensor): The value at which to evaluate the CDF.
        
        Returns:
            torch.Tensor: The probability that a random variable from the distribution is less than or equal to `value`.
        """
        return 0.5 * (1 + torch.erf((value - self.loc) / (self.scale * math.sqrt(2))))

    def expand(self, batch_shape: torch.Size, _instance=None) -> "Normal":
        """
        Expands the distribution parameters to a new batch shape.

        Args:
            batch_shape (torch.Size): Desired batch size for the expanded distribution.
            _instance (Optional): Instance to check for validity.

        Returns:
            Normal: A new Normal distribution with parameters expanded to the specified batch shape.
        """
        new = self._get_checked_instance(Normal, _instance)
        batch_shape = torch.Size(batch_shape)
        new.loc = self.loc.expand(batch_shape)
        new.scale = self.scale.expand(batch_shape)
        super(Normal, new).__init__(batch_shape, validate_args=False)
        new._validate_args = self._validate_args
        return new

    def icdf(self, value: torch.Tensor) -> torch.Tensor:
        """
        Computes the inverse cumulative distribution function (quantile function) at a given value.

        Args:
            value (torch.Tensor): The probability value at which to evaluate the inverse CDF.
        
        Returns:
            torch.Tensor: The quantile corresponding to `value`.
        """
        return self.loc + self.scale * torch.erfinv(2 * value - 1) * math.sqrt(2)

    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        """
        Computes the log probability density of the distribution at a given value.

        Args:
            value (torch.Tensor): The value at which to evaluate the log probability.
        
        Returns:
            torch.Tensor: The log probability density at `value`.
        """
        var = self.scale**2
        log_scale = self.scale.log() if not isinstance(self.scale, Real) else math.log(self.scale)
        return -((value - self.loc) ** 2) / (2 * var) - log_scale - math.log(math.sqrt(2 * math.pi))

    def _transform(self, z: torch.Tensor) -> torch.Tensor:
        """
        Transforms an input tensor `z` to a standardized form based on the mean and scale.

        Args:
            z (torch.Tensor): Input tensor to transform.
        
        Returns:
            torch.Tensor: The transformed tensor, representing the standardized normal form.
        """
        return (z - self.loc) / self.scale

    def _d_transform_d_z(self) -> torch.Tensor:
        """
        Computes the derivative of the transform function with respect to `z`.

        Returns:
            torch.Tensor: The reciprocal of the scale, representing the gradient for reparameterization.
        """
        return 1 / self.scale

    def sample(self, sample_shape: torch.Size = torch.Size()) -> torch.Tensor:
        """
        Generates a sample from the Normal distribution using `torch.normal`.
        
        Args:
            sample_shape (torch.Size): Shape of the sample to generate.
        
        Returns:
            torch.Tensor: A tensor with samples from the Normal distribution, detached from the computation graph.
        """
        shape = self._extended_shape(sample_shape)
        with torch.no_grad():
            return torch.normal(self.loc.expand(shape), self.scale.expand(shape))

    def rsample(self) -> torch.Tensor:
        """
        Generates a reparameterized sample from the Normal distribution, enabling gradient backpropagation.

        Returns:
            torch.Tensor: A tensor containing a reparameterized sample, useful for gradient-based optimization.
        """
        # Sample a point from the distribution
        x = self.sample()
        # Transform the sample to standard normal form
        transform = self._transform(x)
        # Compute a surrogate value for backpropagation
        surrogate_x = -transform / self._d_transform_d_z().detach()
        # Return the sample with gradient tracking enabled
        return x + (surrogate_x - surrogate_x.detach())
        

class MixtureSameFamily(torch.distributions.MixtureSameFamily):
    """
    MixtureSameFamily is a class that represents a mixture of distributions
    from the same family, supporting reparameterized sampling for gradient-based optimization.
    """

    has_rsample = True

    def __init__(self, *args, **kwargs) -> None:
        """
        Initializes the MixtureSameFamily distribution and checks if the component distributions
        support reparameterized sampling (required for `rsample`).

        Raises:
            ValueError: If the component distributions do not support reparameterized sampling.
        """
        super().__init__(*args, **kwargs)
        if not self._component_distribution.has_rsample:
            raise ValueError("Cannot reparameterize a mixture of non-reparameterizable components.")

        # Define a list of discrete distributions for checking in `_log_cdf`
        self.discrete_distributions: List[Distribution] = [
            Bernoulli,
            Binomial,
            ContinuousBernoulli,
            Geometric,
            NegativeBinomial,
            RelaxedBernoulli,
        ]

    def rsample(self, sample_shape: torch.Size = torch.Size()) -> torch.Tensor:
        """
        Generates a reparameterized sample from the mixture of distributions.

        This method generates a sample, applies a distributional transformation,
        and computes a surrogate sample to enable gradient flow during optimization.

        Args:
            sample_shape (torch.Size): The shape of the sample to generate.

        Returns:
            torch.Tensor: A reparameterized sample with gradients enabled.
        """
        # Generate a sample from the mixture distribution
        x = self.sample(sample_shape=sample_shape)
        event_size = math.prod(self.event_shape)

        if event_size != 1:
            # For multi-dimensional events, use reshaped distributional transformations
            def reshaped_dist_trans(input_x: torch.Tensor) -> torch.Tensor:
                return torch.reshape(self._distributional_transform(input_x), (-1, event_size))

            def reshaped_dist_trans_summed(x_2d: torch.Tensor) -> torch.Tensor:
                return torch.sum(reshaped_dist_trans(x_2d), dim=0)

            x_2d = x.reshape((-1, event_size))
            transform_2d = reshaped_dist_trans(x)
            jac = jacobian(reshaped_dist_trans_summed, x_2d).detach().movedim(1, 0)
            surrogate_x_2d = -torch.linalg.solve_triangular(jac.detach(), transform_2d[..., None])
            surrogate_x = surrogate_x_2d.reshape(x.shape)
        else:
            # For one-dimensional events, apply the standard distributional transformation
            transform = self._distributional_transform(x)
            log_prob_x = self.log_prob(x)

            if self._event_ndims > 1:
                log_prob_x = log_prob_x.reshape(log_prob_x.shape + (1,) * self._event_ndims)

            surrogate_x = -transform * torch.exp(-log_prob_x.detach())

        return x + (surrogate_x - surrogate_x.detach())

    def _distributional_transform(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies a distributional transformation to the input sample `x`, using cumulative
        distribution functions (CDFs) and posterior weights.

        Args:
            x (torch.Tensor): The input sample to transform.

        Returns:
            torch.Tensor: The transformed tensor based on the mixture model's CDFs.
        """
        if isinstance(self._component_distribution, torch.distributions.Independent):
            univariate_components = self._component_distribution.base_dist
        else:
            univariate_components = self._component_distribution

        # Expand input tensor and compute log-probabilities in each component
        x = self._pad(x)  # [S, B, 1, E]
        log_prob_x = univariate_components.log_prob(x)  # [S, B, K, E]

        event_size = math.prod(self.event_shape)

        if event_size != 1:
            # CDF transformation for multi-dimensional events
            cumsum_log_prob_x = log_prob_x.reshape(-1, event_size)
            cumsum_log_prob_x = torch.cumsum(cumsum_log_prob_x, dim=-1)
            cumsum_log_prob_x = cumsum_log_prob_x.roll(shifts=1, dims=-1)
            cumsum_log_prob_x[:, 0] = 0
            cumsum_log_prob_x = cumsum_log_prob_x.reshape(log_prob_x.shape)

            logits_mix_prob = self._pad_mixture_dimensions(self._mixture_distribution.logits)
            log_posterior_weights_x = logits_mix_prob + cumsum_log_prob_x

            component_axis = -self._event_ndims - 1
            cdf_x = univariate_components.cdf(x)
            posterior_weights_x = torch.softmax(log_posterior_weights_x, dim=component_axis)
        else:
            # CDF transformation for one-dimensional events
            log_posterior_weights_x = self._mixture_distribution.logits
            component_axis = -self._event_ndims - 1
            cdf_x = univariate_components.cdf(x)
            posterior_weights_x = torch.softmax(log_posterior_weights_x, dim=-1)
            posterior_weights_x = self._pad_mixture_dimensions(posterior_weights_x)

        return torch.sum(posterior_weights_x * cdf_x, dim=component_axis)

    def _log_cdf(self, x: torch.Tensor) -> torch.Tensor:
        """
        Computes the logarithm of the cumulative distribution function (CDF) for the mixture distribution.

        Args:
            x (torch.Tensor): The input tensor for which to compute the log CDF.

        Returns:
            torch.Tensor: The log CDF values.
        """
        x = self._pad(x)
        if callable(getattr(self._component_distribution, "_log_cdf", None)):
            log_cdf_x = self._component_distribution._log_cdf(x)
        else:
            log_cdf_x = torch.log(self._component_distribution.cdf(x))

        if isinstance(self._component_distribution, tuple(self.discrete_distributions)):
            log_mix_prob = torch.sigmoid(self._mixture_distribution.logits)
        else:
            log_mix_prob = F.log_softmax(self._mixture_distribution.logits, dim=-1)

        return torch.logsumexp(log_cdf_x + log_mix_prob, dim=-1)
