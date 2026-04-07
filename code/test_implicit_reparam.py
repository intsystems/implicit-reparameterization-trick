"""
Tests for ImplicitReparam — implicit reparameterization wrapper for arbitrary
factorized distributions (Section 3, Eq. 8 of Figurnov et al., 2019).
"""

import math
import sys
import unittest

import torch

sys.path.append("../src")

from irt.distributions import ImplicitReparam


def implicit_grad(dist, param, f_fn, num_samples=50000):
    """Estimate d/d(param) E[f(z)] via implicit reparam."""
    param.grad = None
    torch.manual_seed(42)
    z = dist.rsample(torch.Size([num_samples]))
    loss = f_fn(z).mean()
    loss.backward()
    return param.grad.clone()


def fd_grad(make_dist_fn, param_val, f_fn, eps=1e-3, num_samples=50000):
    """Estimate d/d(param) E[f(z)] via finite differences."""
    torch.manual_seed(0)
    zp = make_dist_fn(param_val + eps).rsample(torch.Size([num_samples]))
    torch.manual_seed(0)
    zm = make_dist_fn(param_val - eps).rsample(torch.Size([num_samples]))
    return (f_fn(zp).mean() - f_fn(zm).mean()) / (2 * eps)


# ==================== Basic tests ====================


class TestImplicitReparamInit(unittest.TestCase):
    """Test class construction and basic properties."""

    def test_wraps_normal(self):
        loc = torch.tensor(0.0, requires_grad=True)
        base = torch.distributions.Normal(loc, 1.0)
        dist = ImplicitReparam(base)
        self.assertTrue(dist.has_rsample)
        self.assertEqual(dist.batch_shape, base.batch_shape)
        self.assertEqual(dist.event_shape, base.event_shape)

    def test_wraps_exponential(self):
        rate = torch.tensor(2.0, requires_grad=True)
        base = torch.distributions.Exponential(rate)
        dist = ImplicitReparam(base)
        self.assertTrue(dist.has_rsample)
        self.assertEqual(dist.batch_shape, base.batch_shape)

    def test_wraps_independent(self):
        loc = torch.tensor([0.0, 1.0], requires_grad=True)
        scale = torch.tensor([1.0, 2.0], requires_grad=True)
        base = torch.distributions.Independent(torch.distributions.Normal(loc, scale), 1)
        dist = ImplicitReparam(base)
        self.assertEqual(dist.batch_shape, torch.Size())
        self.assertEqual(dist.event_shape, torch.Size([2]))

    def test_rejects_no_cdf(self):
        """Distributions without cdf() should be rejected."""

        class NoCdfDist:
            batch_shape = torch.Size()
            event_shape = torch.Size()
            has_rsample = False

            def sample(self, shape=torch.Size()):
                return torch.randn(shape)

            def log_prob(self, value):
                return torch.zeros_like(value)

        with self.assertRaises(ValueError):
            ImplicitReparam(NoCdfDist())

    def test_wraps_batch(self):
        loc = torch.tensor([0.0, 1.0, 2.0], requires_grad=True)
        base = torch.distributions.Normal(loc, 1.0)
        dist = ImplicitReparam(base)
        self.assertEqual(dist.batch_shape, torch.Size([3]))


# ==================== Sample shape tests ====================


class TestImplicitReparamSamples(unittest.TestCase):
    """Test that rsample produces correct shapes and has gradients."""

    def test_rsample_shape_scalar(self):
        loc = torch.tensor(0.0, requires_grad=True)
        dist = ImplicitReparam(torch.distributions.Normal(loc, 1.0))
        samples = dist.rsample(torch.Size([100]))
        self.assertEqual(samples.shape, torch.Size([100]))
        self.assertTrue(samples.requires_grad)

    def test_rsample_shape_batch(self):
        loc = torch.tensor([0.0, 1.0], requires_grad=True)
        dist = ImplicitReparam(torch.distributions.Normal(loc, 1.0))
        samples = dist.rsample(torch.Size([50]))
        self.assertEqual(samples.shape, torch.Size([50, 2]))
        self.assertTrue(samples.requires_grad)

    def test_rsample_shape_independent(self):
        loc = torch.tensor([0.0, 1.0, 2.0], requires_grad=True)
        base = torch.distributions.Independent(torch.distributions.Normal(loc, 1.0), 1)
        dist = ImplicitReparam(base)
        samples = dist.rsample(torch.Size([50]))
        self.assertEqual(samples.shape, torch.Size([50, 3]))
        self.assertTrue(samples.requires_grad)

    def test_rsample_finite(self):
        loc = torch.tensor(0.0, requires_grad=True)
        dist = ImplicitReparam(torch.distributions.Normal(loc, 1.0))
        samples = dist.rsample(torch.Size([1000]))
        self.assertTrue(torch.isfinite(samples).all())

    def test_gradient_flows(self):
        """Verify gradient flows back to parameters."""
        loc = torch.tensor(0.0, requires_grad=True)
        scale = torch.tensor(1.0, requires_grad=True)
        dist = ImplicitReparam(torch.distributions.Normal(loc, scale))
        z = dist.rsample(torch.Size([100]))
        z.sum().backward()
        self.assertIsNotNone(loc.grad)
        self.assertIsNotNone(scale.grad)

    def test_sample_no_grad(self):
        """sample() should not track gradients."""
        loc = torch.tensor(0.0, requires_grad=True)
        dist = ImplicitReparam(torch.distributions.Normal(loc, 1.0))
        z = dist.sample(torch.Size([10]))
        self.assertFalse(z.requires_grad)


# ==================== Delegation tests ====================


class TestImplicitReparamDelegation(unittest.TestCase):
    """Test that log_prob, cdf, entropy, mean, etc. delegate to base_dist."""

    def setUp(self):
        self.loc = torch.tensor(1.0, requires_grad=True)
        self.scale = torch.tensor(2.0, requires_grad=True)
        self.base = torch.distributions.Normal(self.loc, self.scale)
        self.dist = ImplicitReparam(self.base)

    def test_log_prob(self):
        value = torch.tensor(0.5)
        self.assertTrue(
            torch.allclose(
                self.dist.log_prob(value),
                self.base.log_prob(value),
            )
        )

    def test_cdf(self):
        value = torch.tensor(0.5)
        self.assertTrue(
            torch.allclose(
                self.dist.cdf(value),
                self.base.cdf(value),
            )
        )

    def test_entropy(self):
        self.assertTrue(
            torch.allclose(
                self.dist.entropy(),
                self.base.entropy(),
            )
        )

    def test_mean(self):
        self.assertTrue(torch.allclose(self.dist.mean, self.base.mean))

    def test_variance(self):
        self.assertTrue(torch.allclose(self.dist.variance, self.base.variance))


# ==================== Gradient correctness: known analytical values ====================


class TestNormalImplicitGradients(unittest.TestCase):
    """
    Normal(mu, sigma): d/dmu E[z] = 1, d/dsigma E[z^2] = 2*sigma (for mu=0).
    Verify that ImplicitReparam gives the same gradients as explicit reparam.
    """

    def test_grad_wrt_loc(self):
        loc = torch.tensor(2.0, requires_grad=True)
        dist = ImplicitReparam(torch.distributions.Normal(loc, 1.5))
        grad = implicit_grad(dist, loc, lambda z: z, num_samples=30000)
        self.assertTrue(abs(grad.item() - 1.0) < 0.05, f"d/dloc E[z] = {grad.item():.4f}, expected 1.0")

    def test_grad_wrt_scale(self):
        scale = torch.tensor(2.0, requires_grad=True)
        dist = ImplicitReparam(torch.distributions.Normal(0.0, scale))
        grad = implicit_grad(dist, scale, lambda z: z**2, num_samples=30000)
        expected = 2 * 2.0  # 2*sigma
        self.assertTrue(abs(grad.item() - expected) < 0.3, f"d/dscale E[z^2] = {grad.item():.4f}, expected {expected}")

    def test_matches_explicit_reparam(self):
        """ImplicitReparam(Normal) should give same gradient as Normal.rsample()."""
        loc = torch.tensor(1.0, requires_grad=True)
        scale = torch.tensor(2.0, requires_grad=True)

        # Explicit reparam (PyTorch built-in)
        dist_explicit = torch.distributions.Normal(loc, scale)
        torch.manual_seed(0)
        z_exp = dist_explicit.rsample(torch.Size([10000]))
        z_exp.mean().backward()
        grad_loc_exp = loc.grad.clone()

        # Implicit reparam (our wrapper)
        loc2 = torch.tensor(1.0, requires_grad=True)
        scale2 = torch.tensor(2.0, requires_grad=True)
        dist_implicit = ImplicitReparam(torch.distributions.Normal(loc2, scale2))
        torch.manual_seed(0)
        z_imp = dist_implicit.rsample(torch.Size([10000]))
        z_imp.mean().backward()
        grad_loc_imp = loc2.grad.clone()

        self.assertTrue(
            abs(grad_loc_exp.item() - grad_loc_imp.item()) < 0.05,
            f"Explicit={grad_loc_exp.item():.4f}, Implicit={grad_loc_imp.item():.4f}",
        )


class TestExponentialImplicitGradients(unittest.TestCase):
    """
    Exponential(rate): E[z] = 1/rate, so d/d(rate) E[z] = -1/rate^2.
    """

    def test_grad_wrt_rate(self):
        rate_val = 2.0
        rate = torch.tensor(rate_val, requires_grad=True)
        dist = ImplicitReparam(torch.distributions.Exponential(rate))
        grad = implicit_grad(dist, rate, lambda z: z, num_samples=50000)
        expected = -1.0 / rate_val**2
        self.assertTrue(
            abs(grad.item() - expected) < 0.02, f"d/drate E[z] = {grad.item():.4f}, expected {expected:.4f}"
        )


class TestLaplaceImplicitGradients(unittest.TestCase):
    """
    Laplace(mu, b): d/dmu E[z] = 1.
    """

    def test_grad_wrt_loc(self):
        loc = torch.tensor(1.0, requires_grad=True)
        dist = ImplicitReparam(torch.distributions.Laplace(loc, 2.0))
        grad = implicit_grad(dist, loc, lambda z: z, num_samples=30000)
        self.assertTrue(abs(grad.item() - 1.0) < 0.05, f"d/dloc E[z] = {grad.item():.4f}, expected 1.0")


class TestGumbelImplicitGradients(unittest.TestCase):
    """
    Gumbel(mu, beta): E[z] = mu + beta*gamma, so d/dmu E[z] = 1.
    """

    def test_grad_wrt_loc(self):
        loc = torch.tensor(1.0, requires_grad=True)
        dist = ImplicitReparam(torch.distributions.Gumbel(loc, 2.0))
        grad = implicit_grad(dist, loc, lambda z: z, num_samples=30000)
        self.assertTrue(abs(grad.item() - 1.0) < 0.05, f"d/dloc E[z] = {grad.item():.4f}, expected 1.0")


# ==================== Factorized (Independent) gradient tests ====================


class TestFactorizedImplicitGradients(unittest.TestCase):
    """
    Test implicit reparameterization for factorized multivariate distributions.
    Independent(Normal(loc, scale), 1): each dimension is independent,
    gradient should be computed per-dimension.
    """

    def test_factorized_normal_grad_loc(self):
        """d/d(loc_i) E[z_i] = 1 for each dimension."""
        loc = torch.tensor([0.0, 1.0, 2.0], requires_grad=True)
        scale = torch.tensor([1.0, 2.0, 0.5])
        base = torch.distributions.Independent(torch.distributions.Normal(loc, scale), 1)
        dist = ImplicitReparam(base)

        loc.grad = None
        torch.manual_seed(42)
        z = dist.rsample(torch.Size([30000]))
        z.mean(dim=0).sum().backward()

        for i in range(3):
            self.assertTrue(
                abs(loc.grad[i].item() - 1.0) < 0.05, f"d/dloc[{i}] E[z_{i}] = {loc.grad[i].item():.4f}, expected 1.0"
            )

    def test_factorized_exponential_grad(self):
        """d/d(rate_i) E[z_i] = -1/rate_i^2 for each dimension."""
        rates = torch.tensor([1.0, 2.0, 5.0], requires_grad=True)
        base = torch.distributions.Independent(torch.distributions.Exponential(rates), 1)
        dist = ImplicitReparam(base)

        rates.grad = None
        torch.manual_seed(42)
        z = dist.rsample(torch.Size([50000]))
        z.mean(dim=0).sum().backward()

        for i in range(3):
            expected = -1.0 / rates[i].item() ** 2
            self.assertTrue(
                abs(rates.grad[i].item() - expected) < 0.02,
                f"d/drate[{i}] E[z_{i}] = {rates.grad[i].item():.4f}, expected {expected:.4f}",
            )


# ==================== Finite differences comparison ====================


class TestImplicitVsFiniteDifferences(unittest.TestCase):
    """Compare ImplicitReparam gradients with finite differences for various distributions."""

    def _check_fd(self, name, make_dist_fn, param_val, f_fn, tol=0.1):
        # Implicit
        param = torch.tensor(param_val, requires_grad=True)
        dist = make_dist_fn(param)
        grad_ir = implicit_grad(dist, param, f_fn, num_samples=50000)

        # Finite differences
        grad_fd_val = fd_grad(
            lambda v: make_dist_fn(v),
            param_val,
            f_fn,
            num_samples=50000,
        )
        self.assertTrue(
            abs(grad_ir.item() - grad_fd_val.item()) < tol,
            f"{name}: IR={grad_ir.item():.4f}, FD={grad_fd_val.item():.4f}",
        )

    def test_normal_loc(self):
        self._check_fd(
            "Normal loc",
            lambda p: ImplicitReparam(torch.distributions.Normal(p, 1.0)),
            2.0,
            lambda z: z,
        )

    def test_exponential_rate(self):
        self._check_fd(
            "Exponential rate",
            lambda p: ImplicitReparam(torch.distributions.Exponential(p)),
            3.0,
            lambda z: z,
        )

    def test_laplace_scale(self):
        self._check_fd(
            "Laplace scale",
            lambda p: ImplicitReparam(torch.distributions.Laplace(0.0, p)),
            2.0,
            lambda z: z**2,
        )

    def test_gumbel_scale(self):
        self._check_fd(
            "Gumbel scale",
            lambda p: ImplicitReparam(torch.distributions.Gumbel(0.0, p)),
            2.0,
            lambda z: z**2,
            tol=0.3,
        )


# ==================== Edge cases ====================


class TestImplicitReparamEdgeCases(unittest.TestCase):
    """Numerical stability and edge cases."""

    def test_very_small_scale_normal(self):
        loc = torch.tensor(0.0, requires_grad=True)
        dist = ImplicitReparam(torch.distributions.Normal(loc, 1e-4))
        z = dist.rsample(torch.Size([100]))
        z.sum().backward()
        self.assertTrue(torch.isfinite(loc.grad))

    def test_very_large_rate_exponential(self):
        rate = torch.tensor(1000.0, requires_grad=True)
        dist = ImplicitReparam(torch.distributions.Exponential(rate))
        z = dist.rsample(torch.Size([100]))
        z.sum().backward()
        self.assertTrue(torch.isfinite(rate.grad))

    def test_repeated_rsample(self):
        """Multiple rsample calls should not corrupt state."""
        loc = torch.tensor(0.0, requires_grad=True)
        dist = ImplicitReparam(torch.distributions.Normal(loc, 1.0))
        shapes = []
        for _ in range(5):
            z = dist.rsample(torch.Size([10]))
            shapes.append(z.shape)
        self.assertTrue(all(s == shapes[0] for s in shapes))


if __name__ == "__main__":
    unittest.main()
