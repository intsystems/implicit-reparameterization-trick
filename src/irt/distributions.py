import torch

class torch.distributions.Distribution:
	'''
	The abstract base class for probability distributions, which we inherit from. These methods are implied
	to be implemented for each subclass.
	'''
	def __init__(batch_shape=torch.Size([]), event_shape=torch.Size([])):
		'''
		Basic constructer of distribution.
		'''
	
	@property
	def arg_constraints():
		'''
		Returns a dictionary from argument names to Constraint objects that should
		be satisfied by each argument of this distribution. Args that are not tensors need not appear
		in this dict.
		'''
	
	def cdf(value):
		'''
		Returns the cumulative density/mass function evaluated at value.
		'''
		
	def entropy():
		'''
		Returns entropy of distribution, batched over batch_shape.
		'''

	def enumerate_support(expand=True):
		'''
		Returns tensor containing all values supported by a discrete distribution. The result will
		enumerate over dimension 0, so the shape of the result will be (cardinality,) + batch_shape
		+ event_shape (where event_shape = () for univariate distributions).
		'''
	
	@property
	def mean(expand=True):
		'''
		Returns mean of the distributio.
		'''

	@property
	def mode(expand=True):
		'''
		Returns mean of the distributio.
		'''
	def perplexity():
		'''
		Returns perplexity of distribution, batched over batch_shape.
		'''
	
	def rsample(sample_shape=torch.Size([])):
		'''
		Generates a sample_shape shaped sample or sample_shape shaped batch of samples if the distribution
		parameters are batched.
		'''

	def sample(sample_shape=torch.Size([])):
		'''
		Generates a sample_shape shaped sample or sample_shape shaped batch of reparameterized samples
		if the distribution parameters are batched.
		'''

class torch.distributions.implicit.Normal(Distribution):
	'''
	A Gaussian distribution class with backpropagation capability for the rsample function through IRT.
	'''
	def __init__(mean_matrix, covariance_matrix=None):
		pass

class torch.distributions.implicit.Dirichlet(Distribution):
	'''
	A Dirichlet distribution class with backpropagation capability for the rsample function through IRT.
	'''
	def __init__(concentration, validate_args=None):
		pass
		
class torch.distributions.implicit.Mixture(Distribution):
	'''
	A Mixture of distributions class with backpropagation capability for the rsample function through IRT.
	'''
	def __init__(distributions : List[Distribution]):
		pass

class torch.distributions.implicit.Student(Distribution):
	'''
	A Student's distribution class with backpropagation capability for the rsample function through IRT.
	'''
	def __init__():
		pass

class torch.distributions.implicit.Factorized(Distribution):
	'''
	A class for an arbitrary factorized distribution with backpropagation capability for the rsample
	function through IRT.
	'''
