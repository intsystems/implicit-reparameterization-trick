import unittest
import math
import torch
import sys
sys.path.append('../src')
from irt.distributions import Normal, Gamma, MixtureSameFamily, Beta, Dirichlet, StudentT
from torch.distributions import Categorical, Independent


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

class TestMixtureSameFamily(unittest.TestCase):
    def setUp(self):
        # Use simple distributions for testing. Replace with your desired components
        component_dist = Normal(torch.tensor([0.0, 1.0]).requires_grad_(True), torch.tensor([1.0, 2.0]).requires_grad_(True))
        mixture_dist = Categorical(torch.tensor([0.6, 0.4]))

        self.mixture = MixtureSameFamily(mixture_dist, component_dist)

    def test_rsample_event_size_1(self):
        samples = self.mixture.rsample(sample_shape=torch.Size([10]))
        self.assertEqual(samples.shape, torch.Size([10]))
        self.assertTrue(samples.requires_grad)  # Ensure gradient tracking

    def test_rsample_event_size_greater_than_1(self):
        #Create a mixture with event_size > 1  (e.g., using Independent(Normal(...),1) )
        component_dist = Independent(Normal(torch.tensor([[0.0, 1.0], [2., 3.]]).requires_grad_(True), torch.tensor([[1.0, 2.0], [2., 3.]]).requires_grad_(True)), 1)
        mixture_dist = Categorical(torch.tensor([0.6, 0.4]).requires_grad_(True))
        mixture = MixtureSameFamily(mixture_dist, component_dist)
        samples = mixture.rsample(sample_shape=torch.Size([10]))
        self.assertEqual(samples.shape, torch.Size([10, 2])) # Check shape
        self.assertTrue(samples.requires_grad)  # Ensure gradient tracking

    def test_distributional_transform(self):
        # Test cases for different input shapes and component distributions
        # Add assertions to check the output of _distributional_transform.

        x = torch.randn(10,2)
        transform = self.mixture._distributional_transform(x)
        # ADD ASSERTION HERE.  The transform should be a tensor of the correct shape, depending on event shape of component distribution
        self.assertEqual(transform.shape,torch.Size([10, 2]))


    def test_invalid_component(self):
        # Test with a component distribution that doesn't have rsample
        class NoRsampleDist(object):
            has_rsample = False
        with self.assertRaises(ValueError):
            MixtureSameFamily(Categorical(torch.tensor([0.6, 0.4])), NoRsampleDist())

    def test_log_cdf_multivariate(self):
        # Test with a multivariate component distribution (e.g., Independent(Normal(...),1))
        component_dist = Independent(Normal(loc=torch.zeros(2, 2), scale=torch.ones(2, 2)),1)
        mixture_dist = Categorical(torch.tensor([0.6, 0.4]))
        mixture = MixtureSameFamily(mixture_dist, component_dist)
        x = torch.tensor([[0.5, 1.0], [1.0, 0.5]])
        log_cdf = mixture._log_cdf(x)
        # ADD ASSERTION(S) HERE to check the calculated log_cdf values
        # self.assertTrue(torch.allclose(log_cdf, torch.tensor([expected_value_1, expected_value_2]), atol=1e-4))

class TestBetaDistribution(unittest.TestCase):
    def setUp(self):
        self.concentration1 = torch.tensor([1.0, 2.0], requires_grad=True)
        self.concentration0 = torch.tensor([2.0, 1.0], requires_grad=True)
        self.beta = Beta(self.concentration1, self.concentration0)
        self._dirichlet = Dirichlet(torch.stack([self.concentration1, self.concentration0], -1))  # Initialize _dirichlet

    def test_init(self):
        beta = Beta(torch.tensor(1.0), torch.tensor(2.0))
        self.assertEqual(beta.concentration1, 1.0)
        self.assertEqual(beta.concentration0, 2.0)
        self.assertEqual(beta._gamma1.concentration, 1.0) #Check Gamma parameters
        self.assertEqual(beta._gamma0.concentration, 2.0)


    def test_properties(self):
        self.assertTrue(torch.allclose(self.beta.mean, self.concentration1 / (self.concentration1 + self.concentration0)))
        self.assertTrue(torch.allclose(self.beta.mode, (self.concentration1 - 1) / (self.concentration1 + self.concentration0 - 2)))
        total = self.concentration1 + self.concentration0
        self.assertTrue(torch.allclose(self.beta.variance, self.concentration1 * self.concentration0 / (total.pow(2) * (total + 1))))

    def test_expand(self):
        expanded_beta = self.beta.expand(torch.Size([3, 2]))
        self.assertEqual(expanded_beta.batch_shape, torch.Size([3, 2]))
        self.assertTrue(torch.equal(expanded_beta._gamma1.concentration, self.concentration1.expand([3, 2])))
        self.assertTrue(torch.equal(expanded_beta._gamma0.concentration, self.concentration0.expand([3, 2])))

    def test_rsample(self):
        samples = self.beta.rsample(sample_shape=torch.Size([10]))
        self.assertEqual(samples.shape, torch.Size([10, 2]))
        self.assertTrue(samples.requires_grad)  #check grad tracking

    def test_log_prob(self):
        value = torch.tensor([0.5, 0.7])
        log_prob = self.beta.log_prob(value)
        heads_tails = torch.stack([value, 1.0 - value], -1)
        expected_log_prob = self._dirichlet.log_prob(heads_tails)  #Use initialized _dirichlet
        self.assertTrue(torch.allclose(log_prob, expected_log_prob))

    def test_entropy(self):
        entropy = self.beta.entropy()
        expected_entropy = self._dirichlet.entropy()  #Use initialized _dirichlet
        self.assertTrue(torch.allclose(entropy, expected_entropy))


    def test_natural_params(self):
        natural_params = self.beta._natural_params
        self.assertTrue(torch.equal(natural_params[0], self.concentration1))
        self.assertTrue(torch.equal(natural_params[1], self.concentration0))

    def test_log_normalizer(self):
        x, y = self.beta._natural_params
        log_normalizer = self.beta._log_normalizer(x, y)
        expected_log_normalizer = torch.lgamma(x) + torch.lgamma(y) - torch.lgamma(x + y)
        self.assertTrue(torch.allclose(log_normalizer, expected_log_normalizer))


    def test_invalid_inputs(self):
        with self.assertRaises(ValueError):
            Beta(torch.tensor([-1.0, 1.0]), self.concentration0)  # Negative concentration1
        with self.assertRaises(ValueError):
            Beta(self.concentration1, torch.tensor([-1.0, 1.0]))  # Negative concentration0
        with self.assertRaises(ValueError):
            self.beta.log_prob(torch.tensor([-0.1, 0.5]))  # Value outside [0,1]
        with self.assertRaises(ValueError):
            self.beta.log_prob(torch.tensor([1.1, 0.5]))  

class TestDirichlet(unittest.TestCase):
    def setUp(self):
        self.concentration = torch.tensor([1.0, 2.0, 3.0]).requires_grad_(True)
        self.dirichlet = Dirichlet(self.concentration)

    def test_init(self):
        with self.assertRaises(ValueError):
            Dirichlet(torch.tensor([]), validate_args=True)  # Not enough dimensions
        dirichlet = Dirichlet(self.concentration)
        self.assertTrue(torch.equal(dirichlet.concentration, self.concentration))
        self.assertEqual(dirichlet.batch_shape, torch.Size())
        self.assertEqual(dirichlet.event_shape, torch.Size([3]))

    def test_expand(self):
        expanded_dirichlet = self.dirichlet.expand(torch.Size([2, 3]))
        self.assertEqual(expanded_dirichlet.batch_shape, torch.Size([2, 3]))
        self.assertEqual(expanded_dirichlet.event_shape, torch.Size([3]))
        self.assertTrue(torch.equal(expanded_dirichlet.concentration, self.concentration.expand(torch.Size([2, 3, 3]))))


    def test_log_prob(self):
        value = torch.tensor([0.2, 0.3, 0.5])
        log_prob = self.dirichlet.log_prob(value)
        expected_log_prob = (
            torch.xlogy(self.concentration - 1.0, value).sum(-1)
            + torch.lgamma(self.concentration.sum(-1))
            - torch.lgamma(self.concentration).sum(-1)
        )
        self.assertTrue(torch.allclose(log_prob, expected_log_prob))

    def test_mean(self):
        mean = self.dirichlet.mean
        expected_mean = self.concentration / self.concentration.sum(-1, True)
        self.assertTrue(torch.allclose(mean, expected_mean))

    def test_mode(self):
        mode = self.dirichlet.mode
        concentrationm1 = (self.concentration - 1).clamp(min=0.0)
        expected_mode = concentrationm1 / concentrationm1.sum(-1, True)
        self.assertTrue(torch.allclose(mode, expected_mode))

    def test_variance(self):
        variance = self.dirichlet.variance
        con0 = self.concentration.sum(-1, True)
        expected_variance = (
            self.concentration
            * (con0 - self.concentration)
            / (con0.pow(2) * (con0 + 1))
        )
        self.assertTrue(torch.allclose(variance, expected_variance))

    def test_entropy(self):
        entropy = self.dirichlet.entropy()
        k = self.concentration.size(-1)
        a0 = self.concentration.sum(-1)
        expected_entropy = (
            torch.lgamma(self.concentration).sum(-1)
            - torch.lgamma(a0)
            - (k - a0) * torch.digamma(a0)
            - ((self.concentration - 1.0) * torch.digamma(self.concentration)).sum(-1)
        )
        self.assertTrue(torch.allclose(entropy, expected_entropy))


    def test_natural_params(self):
        natural_params = self.dirichlet._natural_params
        self.assertTrue(torch.equal(natural_params[0], self.concentration))

    def test_log_normalizer(self):
        log_normalizer = self.dirichlet._log_normalizer(self.concentration)
        expected_log_normalizer = torch.lgamma(self.concentration).sum(-1) - torch.lgamma(self.concentration.sum(-1))
        self.assertTrue(torch.allclose(log_normalizer, expected_log_normalizer))

    def test_invalid_inputs(self):
        with self.assertRaises(ValueError):
            Dirichlet(torch.tensor([1.0, -1.0, 3.0])) #Negative Concentration
        with self.assertRaises(ValueError):
            self.dirichlet.log_prob(torch.tensor([0.2, 0.3, 0.6])) #Values don't sum to 1


class TestStudentT(unittest.TestCase):

    def setUp(self):
        self.df = torch.tensor([3.0, 5.0]).requires_grad_(True)
        self.loc = torch.tensor([1.0, 2.0]).requires_grad_(True)
        self.scale = torch.tensor([0.5, 1.0]).requires_grad_(True)
        self.studentt = StudentT(self.df, self.loc, self.scale, validate_args=True)

    def test_init(self):
        studentt = StudentT(3.0, 1.0, 0.5)
        self.assertEqual(studentt.df, 3.0)
        self.assertEqual(studentt.loc, 1.0)
        self.assertEqual(studentt.scale, 0.5)

    def test_properties(self):
        df = torch.tensor([.3, 2.0])
        loc = torch.tensor([1.0, 2.0])
        scale = torch.tensor([0.5, 1.0])
        studentt = StudentT(df, loc, scale)
        self.assertTrue(torch.equal(studentt.mode, studentt.loc))
        # Check mean (undefined for df <= 1)
        # print(self.studentt.mean[0])
        self.assertTrue(torch.isnan(studentt.mean[0])) #Testing for nan values
        self.assertTrue(torch.allclose(studentt.mean[1], studentt.loc[1])) #Mean should be defined for df > 1
        
        # Check variance (undefined for df <= 1, infinite for 1 < df <= 2)
        self.assertTrue(torch.isnan(studentt.variance[0]))
        self.assertTrue(torch.isinf(studentt.variance[1])) # Should be inf for 1 < df <=2
        self.assertTrue(torch.allclose(studentt.variance[1], (scale[1].pow(2) * df[1] / (df[1] - 2)))) #Should be defined for df > 2


    def test_expand(self):
        expanded_studentt = self.studentt.expand(torch.Size([2, 2]))
        self.assertEqual(expanded_studentt.batch_shape, torch.Size([2, 2]))
        self.assertTrue(torch.equal(expanded_studentt.df, self.df.expand([2, 2])))
        self.assertTrue(torch.equal(expanded_studentt.loc, self.loc.expand([2, 2])))
        self.assertTrue(torch.equal(expanded_studentt.scale, self.scale.expand([2, 2])))


    def test_log_prob(self):
        value = torch.tensor([2.0, 3.0])
        log_prob = self.studentt.log_prob(value)
        y = (value - self.loc) / self.scale
        Z = (
            self.scale.log()
            + 0.5 * self.df.log()
            + 0.5 * math.log(math.pi)
            + torch.lgamma(0.5 * self.df)
            - torch.lgamma(0.5 * (self.df + 1.0))
        )
        expected_log_prob = -0.5 * (self.df + 1.0) * torch.log1p(y**2.0 / self.df) - Z
        self.assertTrue(torch.allclose(log_prob, expected_log_prob))


    def test_entropy(self):
        entropy = self.studentt.entropy()
        lbeta = (
            torch.lgamma(0.5 * self.df)
            + math.lgamma(0.5)
            - torch.lgamma(0.5 * (self.df + 1))
        )
        expected_entropy = (
            self.scale.log()
            + 0.5
            * (self.df + 1)
            * (torch.digamma(0.5 * (self.df + 1)) - torch.digamma(0.5 * self.df))
            + 0.5 * self.df.log()
            + lbeta
        )
        self.assertTrue(torch.allclose(entropy, expected_entropy))


    def test_rsample(self):
        samples = self.studentt.rsample(sample_shape=torch.Size([10]))
        print(samples.shape)
        # print(self.studentt.rsample(sample_shape=torch.Size([2])))
        self.assertEqual(samples.shape, torch.Size([10, 2]))
        self.assertTrue(samples.requires_grad) # Check that gradients are tracked

    def test_invalid_inputs(self):
        with self.assertRaises(ValueError):
            StudentT(torch.tensor([-1.0, 1.0]), self.loc, self.scale)  #Negative df
        with self.assertRaises(ValueError):
            self.studentt.log_prob([1, 2])


if __name__ == "__main__":
    unittest.main()
