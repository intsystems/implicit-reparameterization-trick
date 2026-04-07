# Project Plan

Project for the Bayesian Multimodeling course.

## Scope

Implemented distributions:

| Distribution | Status |
|---|---|
| Normal (Gaussian) | Done |
| Gamma | Done |
| Beta | Done |
| Dirichlet | Done |
| Student's t | Done |
| VonMises | Done |
| MixtureSameFamily | Done |
| ImplicitReparam (arbitrary factorized) | Done |

## Architecture

All distributions inherit from `torch.distributions.Distribution` (or `ExponentialFamily`
where applicable) and implement `rsample()` using the implicit reparameterization trick.

The `ImplicitReparam` wrapper enables reparameterized sampling for any distribution
with a tractable CDF, using the universal standardization function (Eq. 8 from the paper).

## Implementation Plan

| Task | Assignee |
|---|---|
| Normal distribution with rsample | Babkin |
| Dirichlet, Beta, Gamma distributions | Zabarianska, Kreinin, Nikitina |
| Student's t-distribution | Babkin |
| MixtureSameFamily | Kreinin |
| VonMises distribution | Kreinin |
| ImplicitReparam (factorized) | Kreinin |
| Unit tests and gradient verification | Kreinin |
| Documentation | Nikitina |
| Blog post | Zabarianska |
| VAE demo on MNIST | Kreinin |
