import unittest
import math
import torch
from torch.distributions import constraints
from torch.distributions.exp_family import ExponentialFamily
from numbers import Number
from irt.distributions import Normal, Gamma


class TestNormalDistribution(unittest.TestCase):
    def setUp(self):
        self.loc = torch.tensor([0.0, 1.0]).requires_grad_(True)
        self.scale = torch.tensor([1.0, 2.0]).requires_grad_(True)
        self.normal = Normal(self.loc, self.scale)

    def test_init(self):
        normal = Normal(0.0, 1.0)
        self.assertEqual(normal.loc, 0.0)
        self.assertEqual(normal.scale, 1.0)
        self.assertEqual(normal.batch_shape, torch.Size())
        
        normal = Normal(torch.tensor([0.0, 1.0]), torch.tensor([1.0, 2.0]))
        self.assertTrue(torch.equal(normal.loc, torch.tensor([0.0, 1.0])))
        self.assertTrue(torch.equal(normal.scale, torch.tensor([1.0, 2.0])))
        self.assertEqual(normal.batch_shape, torch.Size([2]))

    def test_properties(self):
        self.assertTrue(torch.equal(self.normal.mean, self.loc))
        self.assertTrue(torch.equal(self.normal.mode, self.loc))
        self.assertTrue(torch.equal(self.normal.stddev, self.scale))
        self.assertTrue(torch.equal(self.normal.variance, self.scale**2))

    def test_entropy(self):
        entropy = self.normal.entropy()
        expected_entropy = 0.5 + 0.5 * math.log(2 * math.pi) + torch.log(self.scale)
        self.assertTrue(torch.allclose(entropy, expected_entropy))

    def test_cdf(self):
        value = torch.tensor([0.0, 2.0])
        cdf = self.normal.cdf(value)
        expected_cdf = 0.5 * (1 + torch.erf((value - self.loc) / (self.scale * math.sqrt(2))))
        self.assertTrue(torch.allclose(cdf, expected_cdf))

    def test_expand(self):
        expanded_normal = self.normal.expand(torch.Size([3, 2]))
        self.assertEqual(expanded_normal.batch_shape, torch.Size([3, 2]))
        self.assertTrue(torch.equal(expanded_normal.loc, self.loc.expand([3, 2])))
        self.assertTrue(torch.equal(expanded_normal.scale, self.scale.expand([3, 2])))

    def test_icdf(self):
        value = torch.tensor([0.2, 0.8])
        icdf = self.normal.icdf(value)
        expected_icdf = self.loc + self.scale * torch.erfinv(2 * value - 1) * math.sqrt(2)
        self.assertTrue(torch.allclose(icdf, expected_icdf))

    def test_log_prob(self):
        value = torch.tensor([0.0, 2.0])
        log_prob = self.normal.log_prob(value)
        var = self.scale**2
        log_scale = self.scale.log()
        expected_log_prob = -((value - self.loc) ** 2) / (2 * var) - log_scale - math.log(math.sqrt(2 * math.pi))
        self.assertTrue(torch.allclose(log_prob, expected_log_prob))

    def test_sample(self):
        samples = self.normal.sample(sample_shape=torch.Size([100]))
        self.assertEqual(samples.shape, torch.Size([100, 2]))  # Check shape
        emperic_mean = samples.mean(dim=0)
        self.assertTrue((emperic_mean < self.normal.mean + self.normal.scale).all())
        self.assertTrue((self.normal.mean - self.normal.scale < emperic_mean).all())

    def test_rsample(self):
        samples = self.normal.rsample(sample_shape=torch.Size([10]))
        self.assertEqual(samples.shape, torch.Size([10, 2]))  # Check shape
        self.assertTrue(samples.requires_grad)  # Check gradient tracking


class TestGammaDistribution(unittest.TestCase):
    def setUp(self):
        self.concentration = torch.tensor([1.0, 2.0]).requires_grad_(True)
        self.rate = torch.tensor([1.0, 0.5]).requires_grad_(True)
        self.gamma = Gamma(self.concentration, self.rate)

    def test_init(self):
        gamma = Gamma(1.0, 1.0)
        self.assertEqual(gamma.concentration, 1.0)
        self.assertEqual(gamma.rate, 1.0)
        self.assertEqual(gamma.batch_shape, torch.Size())

        gamma = Gamma(torch.tensor([1.0, 2.0]), torch.tensor([1.0, 0.5]))
        self.assertTrue(torch.equal(gamma.concentration, torch.tensor([1.0, 2.0])))
        self.assertTrue(torch.equal(gamma.rate, torch.tensor([1.0, 0.5])))
        self.assertEqual(gamma.batch_shape, torch.Size([2]))

    def test_properties(self):
        self.assertTrue(torch.allclose(self.gamma.mean, self.concentration / self.rate))
        self.assertTrue(torch.allclose(self.gamma.mode, ((self.concentration - 1) / self.rate).clamp(min=0)))
        self.assertTrue(torch.allclose(self.gamma.variance, self.concentration / self.rate.pow(2)))

    def test_expand(self):
        expanded_gamma = self.gamma.expand(torch.Size([3, 2]))
        self.assertEqual(expanded_gamma.batch_shape, torch.Size([3, 2]))
        self.assertTrue(torch.equal(expanded_gamma.concentration, self.concentration.expand([3, 2])))
        self.assertTrue(torch.equal(expanded_gamma.rate, self.rate.expand([3, 2])))

    def test_rsample(self):
        samples = self.gamma.rsample(sample_shape=torch.Size([10]))
        self.assertEqual(samples.shape, torch.Size([10, 2]))  # Check shape
        self.assertTrue(samples.requires_grad) #Check gradient tracking


    def test_log_prob(self):
        value = torch.tensor([1.0, 2.0])
        log_prob = self.gamma.log_prob(value)
        expected_log_prob = (
            torch.xlogy(self.concentration, self.rate)
            + torch.xlogy(self.concentration - 1, value)
            - self.rate * value
            - torch.lgamma(self.concentration)
        )
        self.assertTrue(torch.allclose(log_prob, expected_log_prob))

    def test_entropy(self):
        entropy = self.gamma.entropy()
        expected_entropy = (
            self.concentration
            - torch.log(self.rate)
            + torch.lgamma(self.concentration)
            + (1.0 - self.concentration) * torch.digamma(self.concentration)
        )
        self.assertTrue(torch.allclose(entropy, expected_entropy))

    def test_natural_params(self):
        natural_params = self.gamma._natural_params
        expected_natural_params = (self.concentration - 1, -self.rate)
        self.assertTrue(torch.equal(natural_params[0], expected_natural_params[0]))
        self.assertTrue(torch.equal(natural_params[1], expected_natural_params[1]))

    def test_log_normalizer(self):
        x, y = self.gamma._natural_params
        log_normalizer = self.gamma._log_normalizer(x, y)
        expected_log_normalizer = torch.lgamma(x + 1) + (x + 1) * torch.log(-y.reciprocal())
        self.assertTrue(torch.allclose(log_normalizer, expected_log_normalizer))

    def test_cdf(self):
        value = torch.tensor([1.0, 2.0])
        cdf = self.gamma.cdf(value)
        expected_cdf = torch.special.gammainc(self.concentration, self.rate * value)
        self.assertTrue(torch.allclose(cdf, expected_cdf))


    def test_invalid_inputs(self):
        with self.assertRaises(ValueError):
            Gamma(torch.tensor([-1.0, 1.0]), self.rate)  # Negative concentration
        with self.assertRaises(ValueError):
            Gamma(self.concentration, torch.tensor([-1.0, 1.0]))  # Negative rate
        with self.assertRaises(ValueError):
            self.gamma.log_prob(torch.tensor([-1.0, 1.0]))  # Negative value

if __name__ == "__main__":
    unittest.main()