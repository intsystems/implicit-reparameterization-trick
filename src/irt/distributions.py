# mypy: allow-untyped-defs
import math
from numbers import Number, Real
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch.autograd.functional import jacobian
from torch.distributions import (
    Bernoulli,
    Binomial,
    ContinuousBernoulli,
    Distribution,
    Geometric,
    NegativeBinomial,
    RelaxedBernoulli,
    constraints,
)
from torch.distributions.exp_family import ExponentialFamily
from torch.distributions.utils import broadcast_all, lazy_property
from torch.types import _size

default_size = torch.Size()


class Beta(ExponentialFamily):
    r"""
    Beta distribution parameterized by :attr:`concentration1` and :attr:`concentration0`.

    Example::

        >>> m = Beta(torch.tensor([0.5]), torch.tensor([0.5]))
        >>> m.sample()
        tensor([0.1046])

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

    def __init__(
        self, concentration1: torch.Tensor, concentration0: torch.Tensor, validate_args: Optional[bool] = None
    ) -> None:
        """
        Initializes the Beta distribution with the given concentration parameters.

        Args:
            concentration1: First concentration parameter (alpha).
            concentration0: Second concentration parameter (beta).
            validate_args: If True, validates the distribution's parameters.
        """
        self.concentration1 = concentration1
        self.concentration0 = concentration0
        self._gamma1 = Gamma(self.concentration1, torch.ones_like(concentration1), validate_args=validate_args)
        self._gamma0 = Gamma(self.concentration0, torch.ones_like(concentration0), validate_args=validate_args)
        self._dirichlet = Dirichlet(torch.stack([self.concentration1, self.concentration0], -1))

        super().__init__(self._gamma0._batch_shape, validate_args=validate_args)

    def expand(self, batch_shape: torch.Size, _instance: Optional["Beta"] = None) -> "Beta":
        """
        Expands the Beta distribution to a new batch shape.

        Args:
            batch_shape: Desired batch shape.
            _instance: Instance to validate.

        Returns:
            A new Beta distribution instance with expanded parameters.
        """
        new = self._get_checked_instance(Beta, _instance)
        batch_shape = torch.Size(batch_shape)
        new._gamma1 = self._gamma1.expand(batch_shape)
        new._gamma0 = self._gamma0.expand(batch_shape)
        super(Beta, new).__init__(batch_shape, validate_args=False)
        new._validate_args = self._validate_args
        return new

    @property
    def mean(self) -> torch.Tensor:
        """
        Computes the mean of the Beta distribution.

        Returns:
            torch.Tensor: Mean of the distribution.
        """
        return self.concentration1 / (self.concentration1 + self.concentration0)

    @property
    def mode(self) -> torch.Tensor:
        """
        Computes the mode of the Beta distribution.

        Returns:
            torch.Tensor: Mode of the distribution.
        """
        return (self.concentration1 - 1) / (self.concentration1 + self.concentration0 - 2)

    @property
    def variance(self) -> torch.Tensor:
        """
        Computes the variance of the Beta distribution.

        Returns:
            torch.Tensor: Variance of the distribution.
        """
        total = self.concentration1 + self.concentration0
        return self.concentration1 * self.concentration0 / (total.pow(2) * (total + 1))

    def rsample(self, sample_shape: _size = ()) -> torch.Tensor:
        """
        Generates a reparameterized sample from the Beta distribution.

        Args:
            sample_shape (_size): Shape of the sample.

        Returns:
            torch.Tensor: Sample from the Beta distribution.
        """
        z1 = self._gamma1.rsample(sample_shape)
        z0 = self._gamma0.rsample(sample_shape)
        return z1 / (z1 + z0)

    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        """
        Computes the log probability density of a value under the Beta distribution.

        Args:
            value: Value to evaluate.

        Returns:
            Log probability of the value.
        """
        if self._validate_args:
            self._validate_sample(value)
        heads_tails = torch.stack([value, 1.0 - value], -1)
        return self._dirichlet.log_prob(heads_tails)

    def entropy(self) -> torch.Tensor:
        """
        Computes the entropy of the Beta distribution.

        Returns:
            Entropy of the distribution.
        """
        return self._dirichlet.entropy()

    @property
    def _natural_params(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns the natural parameters of the distribution.

        Returns:
            Natural parameters.
        """
        return self.concentration1, self.concentration0

    def _log_normalizer(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Computes the log normalizer for the natural parameters.

        Args:
            x: Parameter 1.
            y: Parameter 2.

        Returns:
            Log normalizer value.
        """
        return torch.lgamma(x) + torch.lgamma(y) - torch.lgamma(x + y)


class Dirichlet(ExponentialFamily):
    """
    Dirichlet distribution parameterized by a concentration vector.

    The Dirichlet distribution is a multivariate generalization of the Beta distribution. It
    is commonly used in Bayesian statistics, particularly for modeling proportions.
    """

    arg_constraints = {"concentration": constraints.independent(constraints.positive, 1)}
    support = constraints.simplex
    has_rsample = True

    def __init__(self, concentration: torch.Tensor, validate_args: Optional[bool] = None) -> None:
        """
        Initializes the Dirichlet distribution.

        Args:
            concentration: Positive concentration parameter vector (alpha).
            validate_args: If True, validates the distribution's parameters.
        """
        if torch.numel(concentration) < 1:
            raise ValueError("`concentration` parameter must be at least one-dimensional.")
        self.concentration = concentration
        self.gamma = Gamma(self.concentration, torch.ones_like(self.concentration))
        batch_shape, event_shape = concentration.shape[:-1], concentration.shape[-1:]
        super().__init__(batch_shape, event_shape, validate_args=validate_args)

    @property
    def mean(self) -> torch.Tensor:
        """
        Computes the mean of the Dirichlet distribution.

        Returns:
            Mean vector, calculated as `concentration / concentration.sum(-1, keepdim=True)`.
        """
        return self.concentration / self.concentration.sum(-1, keepdim=True)

    @property
    def mode(self) -> torch.Tensor:
        """
        Computes the mode of the Dirichlet distribution.

        Note:
            - The mode is defined only when all concentration values are > 1.
            - For concentrations ≤ 1, the mode vector is clamped to enforce positivity.

        Returns:
            Mode vector.
        """
        concentration_minus_one = (self.concentration - 1).clamp(min=0.0)
        mode = concentration_minus_one / concentration_minus_one.sum(-1, keepdim=True)
        mask = (self.concentration < 1).all(dim=-1)
        mode[mask] = F.one_hot(mode[mask].argmax(dim=-1), concentration_minus_one.shape[-1]).to(mode)
        return mode

    @property
    def variance(self) -> torch.Tensor:
        """
        Computes the variance of the Dirichlet distribution.

        Returns:
            Variance vector for each component.
        """
        total_concentration = self.concentration.sum(-1, keepdim=True)
        return (
            self.concentration
            * (total_concentration - self.concentration)
            / (total_concentration.pow(2) * (total_concentration + 1))
        )

    def rsample(self, sample_shape: _size = ()) -> torch.Tensor:
        """
        Generates a reparameterized sample from the Dirichlet distribution.

        Args:
            sample_shape (_size): Desired sample shape.

        Returns:
            torch.Tensor: A reparameterized sample.
        """
        z = self.gamma.rsample(sample_shape)  # Sample from underlying Gamma distribution

        return z / torch.sum(z, dim=-1, keepdims=True)

    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        """
        Computes the log probability density for a given value.

        Args:
            value (torch.Tensor): Value to evaluate the log probability at.

        Returns:
            torch.Tensor: Log probability density of the value.
        """
        if self._validate_args:
            self._validate_sample(value)
        return (
            torch.xlogy(self.concentration - 1.0, value).sum(-1)
            + torch.lgamma(self.concentration.sum(-1))
            - torch.lgamma(self.concentration).sum(-1)
        )

    def entropy(self) -> torch.Tensor:
        """
        Computes the entropy of the Dirichlet distribution.

        Returns:
            torch.Tensor: Entropy of the distribution.
        """
        k = self.concentration.size(-1)
        total_concentration = self.concentration.sum(-1)
        return (
            torch.lgamma(self.concentration).sum(-1)
            - torch.lgamma(total_concentration)
            - (k - total_concentration) * torch.digamma(total_concentration)
            - ((self.concentration - 1.0) * torch.digamma(self.concentration)).sum(-1)
        )

    def expand(self, batch_shape: torch.Size, _instance: Optional["Dirichlet"] = None) -> "Dirichlet":
        """
        Expands the distribution parameters to a new batch shape.

        Args:
            batch_shape (torch.Size): Desired batch shape.
            _instance (Optional): Instance to validate.

        Returns:
            A new Dirichlet distribution instance with expanded parameters.
        """
        new = self._get_checked_instance(Dirichlet, _instance)
        batch_shape = torch.Size(batch_shape)
        new.concentration = self.concentration.expand(batch_shape + self.event_shape)
        super(Dirichlet, new).__init__(batch_shape, self.event_shape, validate_args=False)
        new._validate_args = self._validate_args
        return new

    @property
    def _natural_params(self) -> tuple:
        """
        Returns the natural parameters of the distribution.

        Returns:
            tuple: Natural parameter tuple `(concentration,)`.
        """
        return (self.concentration,)

    def _log_normalizer(self, x: torch.Tensor) -> torch.Tensor:
        """
        Computes the log normalizer for the natural parameters.

        Args:
            x (torch.Tensor): Natural parameter.

        Returns:
            torch.Tensor: Log normalizer value.
        """
        return x.lgamma().sum(-1) - torch.lgamma(x.sum(-1))


class StudentT(Distribution):
    """
    Student's t-distribution parameterized by degrees of freedom (df), location (loc), and scale (scale).

    This distribution is commonly used for robust statistical modeling, particularly when the data
    may have outliers or heavier tails than a Normal distribution.
    """

    arg_constraints = {
        "df": constraints.positive,
        "loc": constraints.real,
        "scale": constraints.positive,
    }
    support = constraints.real
    has_rsample = True

    def __init__(
        self, df: torch.Tensor, loc: float = 0.0, scale: float = 1.0, validate_args: Optional[bool] = None
    ) -> None:
        """
        Initializes the Student's t-distribution.

        Args:
            df (torch.Tensor): Degrees of freedom (must be positive).
            loc (float or torch.Tensor): Location parameter (default: 0.0).
            scale (float or torch.Tensor): Scale parameter (default: 1.0).
            validate_args (Optional[bool]): If True, validates distribution parameters.
        """
        self.df, self.loc, self.scale = broadcast_all(df, loc, scale)
        batch_shape = self.df.size()
        super().__init__(batch_shape, validate_args=validate_args)

    @property
    def mean(self) -> torch.Tensor:
        """
        Computes the mean of the distribution.

        Note: The mean is undefined when `df <= 1`.

        Returns:
            torch.Tensor: Mean of the distribution, or NaN for undefined cases.
        """
        m = self.loc.clone(memory_format=torch.contiguous_format)
        m[self.df <= 1] = float("nan")  # Mean is undefined for df <= 1
        return m

    @property
    def mode(self) -> torch.Tensor:
        """
        Computes the mode of the distribution.

        Returns:
            torch.Tensor: Mode of the distribution, which is equal to `loc`.
        """
        return self.loc

    @property
    def variance(self) -> torch.Tensor:
        """
        Computes the variance of the distribution.

        Note:
            - Variance is infinite for 1 < df <= 2.
            - Variance is undefined (NaN) for df <= 1.

        Returns:
            torch.Tensor: Variance of the distribution, or appropriate values for edge cases.
        """
        m = self.df.clone(memory_format=torch.contiguous_format)
        # Variance for df > 2
        m[self.df > 2] = self.scale[self.df > 2].pow(2) * self.df[self.df > 2] / (self.df[self.df > 2] - 2)
        # Infinite variance for 1 < df <= 2
        m[(self.df <= 2) & (self.df > 1)] = float("inf")
        # Undefined variance for df <= 1
        m[self.df <= 1] = float("nan")
        return m

    def expand(self, batch_shape: torch.Size, _instance: Optional["StudentT"] = None) -> "StudentT":
        """
        Expands the distribution parameters to a new batch shape.

        Args:
            batch_shape (torch.Size): Desired batch size for the expanded distribution.
            _instance (Optional): Instance to validate.

        Returns:
            StudentT: A new StudentT distribution with expanded parameters.
        """
        new = self._get_checked_instance(StudentT, _instance)
        batch_shape = torch.Size(batch_shape)
        new.df = self.df.expand(batch_shape)
        new.loc = self.loc.expand(batch_shape)
        new.scale = self.scale.expand(batch_shape)
        super(StudentT, new).__init__(batch_shape, validate_args=False)
        new._validate_args = self._validate_args
        return new

    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        """
        Computes the log probability density for a given value.

        Args:
            value (torch.Tensor): Value to evaluate the log probability at.

        Returns:
            torch.Tensor: Log probability density of the given value.
        """
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

    def entropy(self) -> torch.Tensor:
        """
        Computes the entropy of the Student's t-distribution.

        Returns:
            torch.Tensor: Entropy of the distribution.
        """
        lbeta = torch.lgamma(0.5 * self.df) + math.lgamma(0.5) - torch.lgamma(0.5 * (self.df + 1))
        return (
            self.scale.log()
            + 0.5 * (self.df + 1) * (torch.digamma(0.5 * (self.df + 1)) - torch.digamma(0.5 * self.df))
            + 0.5 * self.df.log()
            + lbeta
        )

    def _transform(self, z: torch.Tensor) -> torch.Tensor:
        """
        Transforms an input tensor `z` to a standardized form based on the location and scale.

        Args:
            z (torch.Tensor): Input tensor to transform.

        Returns:
            torch.Tensor: Transformed tensor representing the standardized form.
        """
        return (z - self.loc) / self.scale

    def _d_transform_d_z(self) -> torch.Tensor:
        """
        Computes the derivative of the transform function with respect to `z`.

        Returns:
            torch.Tensor: Reciprocal of the scale, representing the gradient for reparameterization.
        """
        return 1 / self.scale

    def rsample(self, sample_shape: _size = default_size) -> torch.Tensor:
        """
        Generates a reparameterized sample from the Student's t-distribution.

        Args:
            sample_shape (_size): Shape of the sample.

        Returns:
            torch.Tensor: Reparameterized sample, enabling gradient tracking.
        """
        self.loc = self.loc.expand(self._extended_shape(sample_shape))
        self.scale = self.scale.expand(self._extended_shape(sample_shape))
        gamma_samples = Gamma(self.df * 0.5, self.df * 0.5).rsample(sample_shape)
        normal_samples = Normal(torch.zeros(gamma_samples.shape), torch.ones(gamma_samples.shape)).sample()
        
        # Sample from Normal distribution (shape must match after broadcasting)
        x = self.loc.detach() + self.scale.detach() * normal_samples * torch.rsqrt(gamma_samples)

        transform = self._transform(x.detach())  # Standardize the sample
        surrogate_x = -transform / self._d_transform_d_z().detach()  # Compute surrogate gradient

        return x + (surrogate_x - surrogate_x.detach())


class Gamma(ExponentialFamily):
    """
    Gamma distribution parameterized by `concentration` (shape) and `rate` (inverse scale).
    The Gamma distribution is often used to model the time until an event occurs,
    and it is a continuous probability distribution defined for non-negative real values.
    """

    arg_constraints = {
        "concentration": constraints.positive,
        "rate": constraints.positive,
    }
    support = constraints.nonnegative
    has_rsample = True
    _mean_carrier_measure = 0

    def __init__(
        self,
        concentration: torch.Tensor,
        rate: torch.Tensor,
        validate_args: Optional[bool] = None,
    ) -> None:
        """
        Initializes the Gamma distribution.

        Args:
            concentration (torch.Tensor): Shape parameter of the distribution (often referred to as alpha).
            rate (torch.Tensor): Rate parameter (inverse of scale, often referred to as beta).
            validate_args (Optional[bool]): If True, validates the distribution's parameters.
        """
        self.concentration, self.rate = broadcast_all(concentration, rate)
        if isinstance(concentration, Number) and isinstance(rate, Number):
            batch_shape = torch.Size()
        else:
            batch_shape = self.concentration.size()
        super().__init__(batch_shape, validate_args=validate_args)

    @property
    def mean(self) -> torch.Tensor:
        """
        Computes the mean of the Gamma distribution.

        Returns:
            torch.Tensor: Mean of the distribution, calculated as `concentration / rate`.
        """
        return self.concentration / self.rate

    @property
    def mode(self) -> torch.Tensor:
        """
        Computes the mode of the Gamma distribution.

        Note:
            - The mode is defined only for `concentration > 1`. For `concentration <= 1`,
              the mode is clamped to 0.

        Returns:
            torch.Tensor: Mode of the distribution.
        """
        return ((self.concentration - 1) / self.rate).clamp(min=0)

    @property
    def variance(self) -> torch.Tensor:
        """
        Computes the variance of the Gamma distribution.

        Returns:
            torch.Tensor: Variance of the distribution, calculated as `concentration / rate^2`.
        """
        return self.concentration / self.rate.pow(2)

    def expand(self, batch_shape: torch.Size, _instance: Optional["Gamma"] = None) -> "Gamma":
        """
        Expands the distribution parameters to a new batch shape.

        Args:
            batch_shape (torch.Size): Desired batch shape.
            _instance (Optional): Instance to validate.

        Returns:
            Gamma: A new Gamma distribution instance with expanded parameters.
        """
        new = self._get_checked_instance(Gamma, _instance)
        batch_shape = torch.Size(batch_shape)
        new.concentration = self.concentration.expand(batch_shape)
        new.rate = self.rate.expand(batch_shape)
        super(Gamma, new).__init__(batch_shape, validate_args=False)
        new._validate_args = self._validate_args
        return new

    def rsample(self, sample_shape: _size = default_size) -> torch.Tensor:
        """
        Generates a reparameterized sample from the Gamma distribution.

        Args:
            sample_shape (_size): Shape of the sample.

        Returns:
            torch.Tensor: A reparameterized sample.
        """
        shape = self._extended_shape(sample_shape)
        concentration = self.concentration.expand(shape)
        rate = self.rate.expand(shape)

        # Generate a sample using the underlying C++ implementation for efficiency
        value = torch._standard_gamma(concentration) / rate.detach()

        # Detach u for surrogate computation
        u = value.detach() * rate.detach() / rate
        value = value + (u - u.detach())

        # Ensure numerical stability for gradients
        value.detach().clamp_(min=torch.finfo(value.dtype).tiny)
        return value

    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        """
        Computes the log probability density for a given value.

        Args:
            value (torch.Tensor): Value to evaluate the log probability at.

        Returns:
            torch.Tensor: Log probability density of the given value.
        """
        value = torch.as_tensor(value, dtype=self.rate.dtype, device=self.rate.device)
        if self._validate_args:
            self._validate_sample(value)
        return (
            torch.xlogy(self.concentration, self.rate)
            + torch.xlogy(self.concentration - 1, value)
            - self.rate * value
            - torch.lgamma(self.concentration)
        )

    def entropy(self) -> torch.Tensor:
        """
        Computes the entropy of the Gamma distribution.

        Returns:
            torch.Tensor: Entropy of the distribution.
        """
        return (
            self.concentration
            - torch.log(self.rate)
            + torch.lgamma(self.concentration)
            + (1.0 - self.concentration) * torch.digamma(self.concentration)
        )

    @property
    def _natural_params(self) -> tuple:
        """
        Returns the natural parameters of the distribution.

        Returns:
            tuple: Tuple of natural parameters `(concentration - 1, -rate)`.
        """
        return self.concentration - 1, -self.rate

    def _log_normalizer(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Computes the log normalizer for the natural parameters.

        Args:
            x (torch.Tensor): First natural parameter.
            y (torch.Tensor): Second natural parameter.

        Returns:
            torch.Tensor: Log normalizer value.
        """
        return torch.lgamma(x + 1) + (x + 1) * torch.log(-y.reciprocal())

    def cdf(self, value: torch.Tensor) -> torch.Tensor:
        """
        Computes the cumulative distribution function (CDF) for the Gamma distribution.

        Args:
            value (torch.Tensor): Value to evaluate the CDF at.

        Returns:
            torch.Tensor: CDF of the given value.
        """
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

    def expand(self, batch_shape: torch.Size, _instance: Optional["Normal"] = None) -> "Normal":
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

    def sample(self, sample_shape: torch.Size = default_size) -> torch.Tensor:
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

    def rsample(self, sample_shape: _size = default_size) -> torch.Tensor:
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
    Represents a mixture of distributions from the same family.
    Supporting reparameterized sampling for gradient-based optimization.
    """

    has_rsample = True

    def __init__(self, *args, **kwargs) -> None:
        """
        Initializes the MixtureSameFamily distribution and checks if the component distributions.
        Support reparameterized sampling (required for `rsample`).

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

    def rsample(self, sample_shape: torch.Size = default_size) -> torch.Tensor:
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
            surrogate_x_2d = -torch.linalg.solve_triangular(jac.detach(), transform_2d[..., None], upper=False)
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
        if isinstance(self._component_distribution, torch.distributions.Independent):
            univariate_components = self._component_distribution.base_dist
        else:
            univariate_components = self._component_distribution

        if callable(getattr(univariate_components, "_log_cdf", None)):
            log_cdf_x = univariate_components._log_cdf(x)
        else:
            log_cdf_x = torch.log(univariate_components.cdf(x))

        if isinstance(univariate_components, tuple(self.discrete_distributions)):
            log_mix_prob = torch.sigmoid(self._mixture_distribution.logits)
        else:
            log_mix_prob = F.log_softmax(self._mixture_distribution.logits, dim=-1)

        return torch.logsumexp(log_cdf_x + log_mix_prob, dim=-1)


def _eval_poly(y: torch.Tensor, coef: torch.Tensor) -> torch.Tensor:
    """
    Evaluate a polynomial at given points.

    Args:
        y: Input tensor.
        coeffs: Polynomial coefficients.

    Returns:
        Evaluated polynomial tensor.
    """
    coef = list(coef)
    result = coef.pop()
    while coef:
        result = coef.pop() + y * result
    return result


_I0_COEF_SMALL = [
    1.0,
    3.5156229,
    3.0899424,
    1.2067492,
    0.2659732,
    0.360768e-1,
    0.45813e-2,
]
_I0_COEF_LARGE = [
    0.39894228,
    0.1328592e-1,
    0.225319e-2,
    -0.157565e-2,
    0.916281e-2,
    -0.2057706e-1,
    0.2635537e-1,
    -0.1647633e-1,
    0.392377e-2,
]
_I1_COEF_SMALL = [
    0.5,
    0.87890594,
    0.51498869,
    0.15084934,
    0.2658733e-1,
    0.301532e-2,
    0.32411e-3,
]
_I1_COEF_LARGE = [
    0.39894228,
    -0.3988024e-1,
    -0.362018e-2,
    0.163801e-2,
    -0.1031555e-1,
    0.2282967e-1,
    -0.2895312e-1,
    0.1787654e-1,
    -0.420059e-2,
]

_COEF_SMALL = [_I0_COEF_SMALL, _I1_COEF_SMALL]
_COEF_LARGE = [_I0_COEF_LARGE, _I1_COEF_LARGE]


def _log_modified_bessel_fn(x: torch.Tensor, order: int = 0) -> torch.Tensor:
    """
    Compute the logarithm of the modified Bessel function of the first kind.

    Args:
        x: Input tensor, must be positive.
        order: Order of the Bessel function (0 or 1).

    Returns:
        Logarithm of the Bessel function.
    """
    if order not in {0, 1}:
        raise ValueError("Order must be 0 or 1.")

    # compute small solution
    y = x / 3.75
    y = y * y
    small = _eval_poly(y, _COEF_SMALL[order])
    if order == 1:
        small = x.abs() * small
    small = small.log()

    # compute large solution
    y = 3.75 / x
    large = x - 0.5 * x.log() + _eval_poly(y, _COEF_LARGE[order]).log()

    result = torch.where(x < 3.75, small, large)
    return result


@torch.jit.script_if_tracing
def _rejection_sample(
    loc: torch.Tensor, concentration: torch.Tensor, proposal_r: torch.Tensor, x: torch.Tensor
) -> torch.Tensor:
    """
    Perform rejection sampling for the von Mises distribution.

    Args:
        loc: Location parameter.
        concentration: Concentration parameter.
        proposal_r: Precomputed proposal parameter.
        x: Tensor to fill with samples.

    Returns:
        Tensor of samples.
    """
    done = torch.zeros(x.shape, dtype=torch.bool, device=loc.device)
    while not done.all():
        u = torch.rand((3,) + x.shape, dtype=loc.dtype, device=loc.device)
        u1, u2, u3 = u.unbind()
        z = torch.cos(math.pi * u1)
        f = (1 + proposal_r * z) / (proposal_r + z)
        c = concentration * (proposal_r - f)
        accept = ((c * (2 - c) - u2) > 0) | ((c / u2).log() + 1 - c >= 0)
        if accept.any():
            x = torch.where(accept, (u3 - 0.5).sign() * f.acos(), x)
            done = done | accept
    return (x + math.pi + loc) % (2 * math.pi) - math.pi


class VonMises(Distribution):
    """Von Mises distribution class for circular data."""

    arg_constraints = {
        "loc": constraints.real,
        "concentration": constraints.positive,
    }
    support = constraints.real
    has_rsample = True

    def __init__(
        self,
        loc: torch.Tensor,
        concentration: torch.Tensor,
        validate_args: bool = None,
    ) -> None:
        """
        Args:
            loc: loc parameter of the distribution.
            concentration: concentration parameter of the distribution.
            validate_args: If True, checks the distribution parameters for validity.
        """
        self.loc, self.concentration = broadcast_all(loc, concentration)
        batch_shape = self.loc.shape
        super().__init__(batch_shape, torch.Size(), validate_args)

    @lazy_property
    @torch.no_grad()
    def _proposal_r(self) -> torch.Tensor:
        """Compute the proposal parameter for sampling."""
        kappa = self._concentration
        tau = 1 + (1 + 4 * kappa**2).sqrt()
        rho = (tau - (2 * tau).sqrt()) / (2 * kappa)
        _proposal_r = (1 + rho**2) / (2 * rho)

        # second order Taylor expansion around 0 for small kappa
        _proposal_r_taylor = 1 / kappa + kappa
        return torch.where(kappa < 1e-5, _proposal_r_taylor, _proposal_r)

    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        """
        Compute the log probability of the given value.

        Args:
            value: Tensor of values.

        Returns:
            Tensor of log probabilities.
        """
        if self._validate_args:
            self._validate_sample(value)
        log_prob = self.concentration * torch.cos(value - self.loc)
        log_prob = log_prob - math.log(2 * math.pi) - _log_modified_bessel_fn(self.concentration, order=0)
        return log_prob

    @lazy_property
    def _loc(self) -> torch.Tensor:
        return self.loc.to(torch.double)

    @lazy_property
    def _concentration(self) -> torch.Tensor:
        return self.concentration.to(torch.double)

    @torch.no_grad()
    def sample(self, sample_shape: _size = default_size) -> torch.Tensor:
        """
        The sampling algorithm for the von Mises distribution is based on the
        following paper: D.J. Best and N.I. Fisher, "Efficient simulation of the
        von Mises distribution." Applied Statistics (1979): 152-157.

        Sampling is always done in double precision internally to avoid a hang
        in _rejection_sample() for small values of the concentration, which
        starts to happen for single precision around 1e-4 (see issue #88443).
        """
        shape = self._extended_shape(sample_shape)
        x = torch.empty(shape, dtype=self._loc.dtype, device=self.loc.device)
        return _rejection_sample(self._loc, self._concentration, self._proposal_r, x).to(self.loc.dtype)

    def rsample(self, sample_shape: _size = default_size) -> torch.Tensor:
        """Generate reparameterized samples from the distribution"""
        shape = self._extended_shape(sample_shape)
        samples = _VonMisesSampler.apply(self.concentration, self._proposal_r, shape)
        samples = samples + self.loc

        # Map the samples to [-pi, pi].
        return samples - 2.0 * torch.pi * torch.round(samples / (2.0 * torch.pi))

    @property
    def mean(self) -> torch.Tensor:
        """Mean of the distribution."""
        return self.loc

    @property
    def variance(self) -> torch.Tensor:
        """Variance of the distribution."""
        return (
            1
            - (
                _log_modified_bessel_fn(self.concentration, order=1)
                - _log_modified_bessel_fn(self.concentration, order=0)
            ).exp()
        )


@torch.jit.script_if_tracing
@torch.no_grad()
def _rejection_rsample(concentration: torch.Tensor, proposal_r: torch.Tensor, shape: torch.Size) -> torch.Tensor:
    """
    Perform rejection sampling to draw samples from the von Mises distribution.

    Args:
        concentration (torch.Tensor): Concentration parameter (kappa) of the distribution.
        proposal_r (torch.Tensor): Proposal distribution parameter.
        shape (torch.Size): Desired shape of the samples.

    Returns:
        torch.Tensor: Samples from the von Mises distribution.
    """
    x = torch.empty(shape, dtype=concentration.dtype, device=concentration.device)
    done = torch.zeros(x.shape, dtype=torch.bool, device=concentration.device)

    while not done.all():
        u = torch.rand((3,) + x.shape, dtype=concentration.dtype, device=concentration.device)
        u1, u2, u3 = u.unbind()
        z = torch.cos(math.pi * u1)
        f = (1 + proposal_r * z) / (proposal_r + z)
        c = concentration * (proposal_r - f)
        accept = ((c * (2 - c) - u2) > 0) | ((c / u2).log() + 1 - c >= 0)
        if accept.any():
            x = torch.where(accept, (u3 - 0.5).sign() * f.acos(), x)
            done = done | accept
    return x


def cosxm1(x: torch.Tensor) -> torch.Tensor:
    """
    Compute cos(x) - 1 using a numerically stable formula.

    Args:
        x (torch.Tensor): Input tensor.

    Returns:
        torch.Tensor: Output tensor, `cos(x) - 1`.
    """
    return -2 * torch.square(torch.sin(x / 2.0))


class _VonMisesSampler(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        concentration: torch.Tensor,
        proposal_r: torch.Tensor,
        shape: torch.Size,
    ) -> torch.Tensor:
        """
        Perform forward sampling using rejection sampling.

        Args:
            ctx (torch.autograd.function.FunctionCtx): Context object for saving tensors.
            concentration (torch.Tensor): Concentration parameter (kappa).
            proposal_r (torch.Tensor): Proposal distribution parameter.
            shape (torch.Size): Desired shape of the samples.

        Returns:
            torch.Tensor: Samples from the von Mises distribution.
        """
        samples = _rejection_rsample(concentration, proposal_r, shape)
        ctx.save_for_backward(concentration, proposal_r, samples)

        return samples

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(
        ctx: torch.autograd.function.FunctionCtx, grad_output: torch.Tensor
    ) -> Tuple[torch.Tensor, None, None]:
        """
        Compute gradients for backward pass using implicit reparameterization.

        Args:
            ctx (torch.autograd.function.FunctionCtx): Context object containing saved tensors.
            grad_output (torch.Tensor): Gradient of the loss with respect to the output.

        Returns:
            Tuple[torch.Tensor, None, None]: Gradients with respect to the input tensors.
        """
        concentration, proposal_r, samples = ctx.saved_tensors

        num_periods = torch.round(samples / (2.0 * torch.pi))
        x_mapped = samples - (2.0 * torch.pi) * num_periods

        # Parameters from the paper
        ck = 10.5
        num_terms = 20

        # Compute series and normal approximation
        cdf_series, dcdf_dconcentration_series = von_mises_cdf_series(x_mapped, concentration, num_terms)
        cdf_normal, dcdf_dconcentration_normal = von_mises_cdf_normal(x_mapped, concentration)
        use_series = concentration < ck
        # cdf = torch.where(use_series, cdf_series, cdf_normal) + num_periods
        dcdf_dconcentration = torch.where(use_series, dcdf_dconcentration_series, dcdf_dconcentration_normal)

        # Compute CDF gradient terms
        inv_prob = torch.exp(concentration * cosxm1(samples)) / (2 * math.pi * torch.special.i0e(concentration))
        grad_concentration = grad_output * (-dcdf_dconcentration / inv_prob)

        return grad_concentration, None, None


def von_mises_cdf_series(
    x: torch.Tensor, concentration: torch.Tensor, num_terms: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute the CDF of the von Mises distribution using a series approximation.

    Args:
        x (torch.Tensor): Input tensor.
        concentration (torch.Tensor): Concentration parameter (kappa).
        num_terms (int): Number of terms in the series.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: CDF and its gradient with respect to concentration.
    """
    vn = torch.zeros_like(x)
    dvn_dconcentration = torch.zeros_like(x)

    n = torch.tensor(num_terms, dtype=x.dtype, device=x.device)
    rn = torch.zeros_like(x)
    drn_dconcentration = torch.zeros_like(x)

    while n > 0:
        denominator = 2.0 * n / concentration + rn
        ddenominator_dk = -2.0 * n / concentration**2 + drn_dconcentration
        rn = 1.0 / denominator
        drn_dconcentration = -ddenominator_dk / denominator**2

        multiplier = torch.sin(n * x) / n + vn
        vn = rn * multiplier
        dvn_dconcentration = drn_dconcentration * multiplier + rn * dvn_dconcentration

        n -= 1

    cdf = 0.5 + x / (2.0 * torch.pi) + vn / torch.pi
    dcdf_dconcentration = dvn_dconcentration / torch.pi

    cdf_clipped = torch.clamp(cdf, 0.0, 1.0)
    dcdf_dconcentration *= (cdf >= 0.0) & (cdf <= 1.0)

    return cdf_clipped, dcdf_dconcentration


def cdf_func(concentration: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """
    Approximate the CDF of the von Mises distribution.

    Args:
        concentration (torch.Tensor): Concentration parameter (kappa).
        x (torch.Tensor): Input tensor.

    Returns:
        torch.Tensor: Approximate CDF values.
    """
    # Calculate the z value based on the approximation
    z = (torch.sqrt(torch.tensor(2.0 / torch.pi)) / torch.special.i0e(concentration)) * torch.sin(0.5 * x)
    # Apply corrections to z to improve the approximation
    z2 = z**2
    z3 = z2 * z
    z4 = z2**2
    c = 24.0 * concentration
    c1 = 56.0

    xi = z - z3 / (((c - 2.0 * z2 - 16.0) / 3.0) - (z4 + (7.0 / 4.0) * z2 + 167.0 / 2.0) / (c - c1 - z2 + 3.0)) ** 2

    # Use the standard normal distribution for the approximation
    distrib = torch.distributions.Normal(
        torch.tensor(0.0, dtype=x.dtype, device=x.device), torch.tensor(1.0, dtype=x.dtype, device=x.device)
    )

    return distrib.cdf(xi)


def von_mises_cdf_normal(x: torch.Tensor, concentration: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute the CDF of the von Mises distribution using a normal approximation.

    Args:
        x (torch.Tensor): Input tensor.
        concentration (torch.Tensor): Concentration parameter (kappa).

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: CDF and its gradient with respect to concentration.
    """
    with torch.enable_grad():
        concentration_ = concentration.detach().clone().requires_grad_(True)
        cdf = cdf_func(concentration_, x)
        cdf.backward(torch.ones_like(cdf))  # Compute gradients
        dcdf_dconcentration = concentration_.grad.clone()  # Copy the gradient
    # Detach gradients to prevent further autograd tracking
    concentration_.grad = None
    return cdf, dcdf_dconcentration
