# Project plan and describtion
> [!NOTE]
> Although we have tried to make the plan final, it is possible that changes may occur.

The outcome of our __Implicit Reparametrization Trick__ project is an extension to the torch.distributions library. It will be called torch.distributions.implicit, and will contain both the distributions implemented in torch.distributions, as well as new ones that are not present there.
## Scope
We plan to implement the following distributions in our library:
- Gaussian normal distribution (*)
- Dirichlet distribution (Beta distributions)
- Sampling from a mixture of distributions
- Sampling from the Student's t-distribution (**)
- Sampling from an arbitrary factorized distribution (***)

(\*) - this distribution is already implemented in torch using the explicit reparameterization trick, we will implement it for comparison
(\*\*) - this distribution is added as a backup, their inclusion is questionable
(\*\*\*) - this distribution is not very clear in implementation, its inclusion is questionable

## Stack
![stack](./images/stack)



## Scheme of classes and examples

Thus, the scope of the library is to implement following classes:
```
class torch.distributions.implicit.normal
class torch.distributions.implicit.normal
```

## Plan of implementation

