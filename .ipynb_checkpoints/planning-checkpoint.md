# Project plan and describtion
> [!NOTE]
> Although we have tried to make the plan final, it is possible that changes may occur.

The outcome of our __Implicit Reparametrization Trick__ project is an extension to the torch.distributions library. It will be called torch.distributions.implicit, and will contain both the distributions implemented in torch.distributions, as well as new ones that are not present there.
## Scope
We plan to implement the following distributions in our library:
- Gaussian normal distribution (*)
- Dirichlet distribution (Beta distributions)(\*)
- Sampling from a mixture of distributions
- Sampling from the Student's t-distribution (**) (\*)
- Sampling from an arbitrary factorized distribution (***)

(\*) - this distribution is already implemented in torch using the explicit reparameterization trick, we will implement it for comparison

(\*\*) - this distribution is added as a backup, their inclusion is questionable

(\*\*\*) - this distribution is not very clear in implementation, its inclusion is questionable

## Stack

We plan to inherit from the torch.distribution.Distribution class, so we need to implement all the methods that are present in that class.

![stack](./images/stack.png)

On this diagram, the elements marked in black already exist. The elements marked in green are planned to be implemented. The elements marked in blue are ones whose implementation is uncertain.

## Scheme of classes and examples

In this section, we present the implementation scheme of the library, showing the classes and their methods that are planned to be implemented. Here we provide a description of the classes and the arguments they accept.
```
class torch.distributions.Distribution:
	'''
	The abstract base class for probability distributions, which we inherit from. These methods are implied
	to be implemented for each subclass.
	'''    
    has_rsample = True
    

    def __init__(batch_shape=torch.Size([]), event_shape=torch.Size([])):
		'''
		Basic constructer of distribution.
		'''

    @property
    def mean(self) -> torch.Tensor:
        pass

    @property
    def mode(self) -> torch.Tensor:
        pass

    @property
    def stddev(self) -> torch.Tensor:
        pass

    @property
    def variance(self) -> torch.Tensor:
        pass

    def entropy(self) -> torch.Tensor:
        pass

    def cdf(self, value: torch.Tensor) -> torch.Tensor:
        pass

    def expand(self, batch_shape: torch.Size, _instance=None) -> "Normal":
        pass

    def icdf(self, value: torch.Tensor) -> torch.Tensor:
        pass

    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        pass

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
```
####  Usage
In this example, we demonstrate the application of our library using a Variational Autoencoder (VAE) model, where the latent layer is modified by a normal distribution.
```
>>> import torch.distributions.implicit as irt
>>> params = Encoder(inputs)
>>> gauss = irt.Normal(*params)
>>> deviated = gauss.rsample()
>>> outputs = Decoder(deviated)
```
In this example, we demonstrate the use of a mixture of distributions using our library.
```
>>> import torch.distributions.implicit as irt
>>> params = Encoder(inputs)
>>> mix = irt.Mixture([irt.Normal(*params), irt.Dirichlet(*params)])
>>> deviated = mix.rsample()
>>> outputs = Decoder(deviated)
```

## Plan of implementation

| Task | Deadline | Assignee |
| ---- | ---- |  ---- |
| Construct a basic example of a Normal class with an rsample method that supports backpropagation. | 22.10 | Babkin |
| Test usability of the library in VAE on MNIST. | 22.10 | - |
| Compare results with explicit reparametrization trick. | 22.10 | - |
| Prepare a draft of documentation. | 28.10 | Nikitina |
| Prepare a draft of blogpost. | 28.10 | Zabarianska |
| Expand the class Normal and prepare distributions Dirichlet and Mixture. | 12.11 | Zabarianska, Kreinin, Nikitina |
| Try to implement class Factorized. | 12.11 | Babkin |
| Implement class Student's in case of failure. | 12.11 | Babkin |
| Wrap project into a library. | 12.11 | Nikitina |
| Write unit tests for different distribution, try to cover all cases. | 12.11 | Kreinin |
| Prepare a demo. | 19.11 | Kreinin |
| Expand documentation for our code. | 19.11 | Nikitina |
| Prepare a blog-post. | 19.11 | Zabarianska | 
