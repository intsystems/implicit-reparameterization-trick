# mypy: allow-untyped-defs
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
from torch.types import _size


class Beta(ExponentialFamily):
    r"""
    Beta distribution parameterized by :attr:`concentration1` and :attr:`concentration0`.

    Example::

        >>> # xdoctest: +IGNORE_WANT("non-deterministic")
        >>> m = Beta(torch.tensor([0.5]), torch.tensor([0.5]))
        >>> m.sample()  # Beta distributed with concentration concentration1 and concentration0
        tensor([ 0.1046])

    Args:
        concentration1 (float or Tensor): 1st concentration parameter of the distribution
            (often referred to as alpha)
        concentration0 (float or Tensor): 2nd concentration parameter of the distribution
            (often referred to as beta)
    """
    arg_constraints = {
        "concentration1": constraints.positive,
        "concentration0": constraints.positive,
    }
    support = constraints.unit_interval
    has_rsample = True

    def __init__(self, concentration1, concentration0, validate_args=None):
        #self.df, self.loc, self.scale = broadcast_all(df, loc, scale)
        self.concentration1 = concentration1
        self.concentration0 = concentration0
        self._gamma1 = Gamma(self.concentration1, torch.ones_like(concentration1), validate_args=validate_args)
        self._gamma0 = Gamma(self.concentration0, torch.ones_like(concentration0), validate_args=validate_args)
        
        super().__init__(self._gamma0._batch_shape, validate_args=validate_args)

    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(Beta, _instance)
        batch_shape = torch.Size(batch_shape)
        new._gamma1 = self._gamma1.expand(batch_shape)
        new._gamma0 = self._gamma0.expand(batch_shape)
        super(Beta, new).__init__(batch_shape, validate_args=False)
        new._validate_args = self._validate_args
        return new

    @property
    def mean(self):
        return self.concentration1 / (self.concentration1 + self.concentration0)

    @property
    def mode(self):
        return (self.concentration1 - 1)/(self.concentration1 + self.concentration0 - 2)

    @property
    def variance(self):
        total = self.concentration1 + self.concentration0
        return self.concentration1 * self.concentration0 / (total.pow(2) * (total + 1))

    def rsample(self, sample_shape: _size = ()) -> torch.Tensor:
        z1 = self._gamma1.rsample(sample_shape)
        z0 = self._gamma0.rsample(sample_shape)
        return z1/(z1+z0)

    def log_prob(self, value):
        if self._validate_args:
            self._validate_sample(value)
        heads_tails = torch.stack([value, 1.0 - value], -1)
        return self._dirichlet.log_prob(heads_tails)

    def entropy(self):
        return self._dirichlet.entropy()

    @property
    def _natural_params(self):
        return (self.concentration1, self.concentration0)

    def _log_normalizer(self, x, y):
        return torch.lgamma(x) + torch.lgamma(y) - torch.lgamma(x + y)


class Dirichlet(ExponentialFamily):
    r"""
    Creates a Dirichlet distribution parameterized by concentration :attr:`concentration`.

    Example::

        >>> # xdoctest: +IGNORE_WANT("non-deterministic")
        >>> m = Dirichlet(torch.tensor([0.5, 0.5]))
        >>> m.sample()  # Dirichlet distributed with concentration [0.5, 0.5]
        tensor([ 0.1046,  0.8954])

    Args:
        concentration (Tensor): concentration parameter of the distribution
            (often referred to as alpha)
    """
    arg_constraints = {
        "concentration": constraints.independent(constraints.positive, 1)
    }
    support = constraints.simplex
    has_rsample = True

    def __init__(self, concentration, validate_args=None):
        if concentration.dim() < 1:
            raise ValueError(
                "`concentration` parameter must be at least one-dimensional."
            )
        self.concentration = concentration
        self.gamma = Gamma(self.concentration, torch.ones_like(self.concentration))
        batch_shape, event_shape = concentration.shape[:-1], concentration.shape[-1:]
        super().__init__(batch_shape, event_shape, validate_args=validate_args)


    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(Dirichlet, _instance)
        batch_shape = torch.Size(batch_shape)
        new.concentration = self.concentration.expand(batch_shape + self.event_shape)
        super(Dirichlet, new).__init__(
            batch_shape, self.event_shape, validate_args=False
        )
        new._validate_args = self._validate_args
        return new


    def rsample(self, sample_shape: _size = ()) -> torch.Tensor:
        z = self.gamma.rsample(sample_shape)
        if len(self.batch_shape) == 0 or len(sample_shape) == 0:
            dim = 0
        else:
            dim = tuple(range(1, z.dim()))
        
        return z*torch.exp(-torch.sum(z, dim=dim))


    def log_prob(self, value):
        if self._validate_args:
            self._validate_sample(value)
        return (
            torch.xlogy(self.concentration - 1.0, value).sum(-1)
            + torch.lgamma(self.concentration.sum(-1))
            - torch.lgamma(self.concentration).sum(-1)
        )


    @property
    def mean(self):
        return self.concentration / self.concentration.sum(-1, True)

    @property
    def mode(self):
        concentrationm1 = (self.concentration - 1).clamp(min=0.0)
        mode = concentrationm1 / concentrationm1.sum(-1, True)
        mask = (self.concentration < 1).all(axis=-1)
        mode[mask] = torch.nn.functional.one_hot(
            mode[mask].argmax(axis=-1), concentrationm1.shape[-1]
        ).to(mode)
        return mode

    @property
    def variance(self):
        con0 = self.concentration.sum(-1, True)
        return (
            self.concentration
            * (con0 - self.concentration)
            / (con0.pow(2) * (con0 + 1))
        )


    def entropy(self):
        k = self.concentration.size(-1)
        a0 = self.concentration.sum(-1)
        return (
            torch.lgamma(self.concentration).sum(-1)
            - torch.lgamma(a0)
            - (k - a0) * torch.digamma(a0)
            - ((self.concentration - 1.0) * torch.digamma(self.concentration)).sum(-1)
        )


    @property
    def _natural_params(self):
        return (self.concentration,)

    def _log_normalizer(self, x):
        return x.lgamma().sum(-1) - torch.lgamma(x.sum(-1))



class StudentT(Distribution):
    arg_constraints = {
        "df": constraints.positive,
        "loc": constraints.real,
        "scale": constraints.positive,
    }
    support = constraints.real
    has_rsample = True

    def __init__(self, df, loc=0.0, scale=1.0, validate_args=None):
        self.df, self.loc, self.scale = broadcast_all(df, loc, scale)
        self.gamma = Gamma(self.df*0.5, self.df*0.5)
        batch_shape = self.df.size()
        super().__init__(batch_shape, validate_args=validate_args)
        
    @property
    def mean(self):
        m = self.loc.clone(memory_format=torch.contiguous_format)
        m[self.df <= 1] = nan
        return m

    @property
    def mode(self):
        return self.loc

    @property
    def variance(self):
        m = self.df.clone(memory_format=torch.contiguous_format)
        m[self.df > 2] = (
            self.scale[self.df > 2].pow(2)
            * self.df[self.df > 2]
            / (self.df[self.df > 2] - 2)
        )
        m[(self.df <= 2) & (self.df > 1)] = inf
        m[self.df <= 1] = nan
        return m

    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(StudentT, _instance)
        batch_shape = torch.Size(batch_shape)
        new.df = self.df.expand(batch_shape)
        new.loc = self.loc.expand(batch_shape)
        new.scale = self.scale.expand(batch_shape)
        super(StudentT, new).__init__(batch_shape, validate_args=False)
        new._validate_args = self._validate_args
        return new


    def log_prob(self, value):
        if self._validate_args:
            self._validate_sample(value)
        y = (value - self.loc) / self.scale
        Z = (
            self.scale.log()
            + 0.5 * self.df.log()
            + 0.5 * math.log(math.pi)
            + torch.lgamma(0.5 * self.df)
            - torch.lgamma(0.5 * (self.df + 1.0))
        )
        return -0.5 * (self.df + 1.0) * torch.log1p(y**2.0 / self.df) - Z


    def entropy(self):
        lbeta = (
            torch.lgamma(0.5 * self.df)
            + math.lgamma(0.5)
            - torch.lgamma(0.5 * (self.df + 1))
        )
        return (
            self.scale.log()
            + 0.5
            * (self.df + 1)
            * (torch.digamma(0.5 * (self.df + 1)) - torch.digamma(0.5 * self.df))
            + 0.5 * self.df.log()
            + lbeta
        )

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

    def rsample(self, sample_shape: _size = torch.Size()) -> torch.Tensor:        
        shape = self._extended_shape(sample_shape)
        
        sigma = self.gamma.rsample(shape)
        x = self.loc.detach() + self.scale.detach() * Normal(0, sigma).rsample(shape)

        transform = self._transform(x.detach())
        
        surrogate_x = -transform / self._d_transform_d_z().detach()
        
        return x + (surrogate_x - surrogate_x.detach())

class Gamma(ExponentialFamily):
    r"""
    Creates a Gamma distribution parameterized by shape :attr:`concentration` and :attr:`rate`.

    Example::

        >>> # xdoctest: +IGNORE_WANT("non-deterministic")
        >>> m = Gamma(torch.tensor([1.0]), torch.tensor([1.0]))
        >>> m.sample()  # Gamma distributed with concentration=1 and rate=1
        tensor([ 0.1046])

    Args:
        concentration (float or Tensor): shape parameter of the distribution
            (often referred to as alpha)
        rate (float or Tensor): rate = 1 / scale of the distribution
            (often referred to as beta)
    """
    arg_constraints = {
        "concentration": constraints.positive,
        "rate": constraints.positive,
    }
    support = constraints.nonnegative
    has_rsample = True
    _mean_carrier_measure = 0

    @property
    def mean(self):
        return self.concentration / self.rate

    @property
    def mode(self):
        return ((self.concentration - 1) / self.rate).clamp(min=0)

    @property
    def variance(self):
        return self.concentration / self.rate.pow(2)

    def __init__(self, concentration, rate, validate_args=None):
        self.concentration, self.rate = broadcast_all(concentration, rate)
        if isinstance(concentration, Number) and isinstance(rate, Number):
            batch_shape = torch.Size()
        else:
            batch_shape = self.concentration.size()
        super().__init__(batch_shape, validate_args=validate_args)

    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(Gamma, _instance)
        batch_shape = torch.Size(batch_shape)
        new.concentration = self.concentration.expand(batch_shape)
        new.rate = self.rate.expand(batch_shape)
        super(Gamma, new).__init__(batch_shape, validate_args=False)
        new._validate_args = self._validate_args
        return new


    def rsample(self, sample_shape: _size = torch.Size()) -> torch.Tensor:
        shape = self._extended_shape(sample_shape)
        concentration = self.concentration.expand(shape)
        rate = self.rate.expand(shape)
        
        ##################################################################################################################
        ### U can think that this method is total shit                                                                 ###
        ### BUT you MUST calculate for gamma (Of course we can do it, I can give you version of it.)                   ###
        ### BUT we MUST do it in more efficient way and the most efficient way is to use C++ implementation from Torch.###
        ##################################################################################################################
        value = torch._standard_gamma(self.concentration)/rate.detach()
        u = value.detach() * rate.detach() / rate
        value = value + (u - u.detach())
        
        value.detach().clamp_(
            min=torch.finfo(value.dtype).tiny
        )  # do not record in autograd graph
        return value 


    def log_prob(self, value):
        value = torch.as_tensor(value, dtype=self.rate.dtype, device=self.rate.device)
        if self._validate_args:
            self._validate_sample(value)
        return (
            torch.xlogy(self.concentration, self.rate)
            + torch.xlogy(self.concentration - 1, value)
            - self.rate * value
            - torch.lgamma(self.concentration)
        )


    def entropy(self):
        return (
            self.concentration
            - torch.log(self.rate)
            + torch.lgamma(self.concentration)
            + (1.0 - self.concentration) * torch.digamma(self.concentration)
        )


    @property
    def _natural_params(self):
        return (self.concentration - 1, -self.rate)

    def _log_normalizer(self, x, y):
        return torch.lgamma(x + 1) + (x + 1) * torch.log(-y.reciprocal())

    def cdf(self, value):
        if self._validate_args:
            self._validate_sample(value)
        return torch.special.gammainc(self.concentration, self.rate * value)

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
        
    def rsample(self, sample_shape: _size = torch.Size()) -> torch.Tensor:
        """
        Generates a reparameterized sample from the Normal distribution, enabling gradient backpropagation.

        Returns:
            torch.Tensor: A tensor containing a reparameterized sample, useful for gradient-based optimization.
        """
        # Sample a point from the distribution
        x = self.sample(sample_shape)
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
