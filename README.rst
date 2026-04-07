Implicit Reparameterization Trick
=================================

|test| |docs| |pytorch| |license|

.. |test| image:: https://github.com/intsystems/implicit-reparameterization-trick/actions/workflows/testing.yml/badge.svg
    :target: https://github.com/intsystems/implicit-reparameterization-trick/actions/workflows/testing.yml

.. |docs| image:: https://github.com/intsystems/implicit-reparameterization-trick/actions/workflows/docs.yml/badge.svg
    :target: https://intsystems.github.io/implicit-reparameterization-trick/

.. |pytorch| image:: https://img.shields.io/badge/PyTorch-%3E%3D2.1-EE4C2C?logo=pytorch&logoColor=white
    :target: https://pytorch.org/

.. |license| image:: https://img.shields.io/badge/License-MIT-blue.svg
    :target: https://opensource.org/licenses/MIT

A PyTorch library implementing implicit reparameterization gradients for continuous
distributions that lack tractable inverse CDFs, based on
`Figurnov et al. (NeurIPS 2018) <https://arxiv.org/abs/1805.08498>`_.

Implemented distributions: Normal, Gamma, Beta, Dirichlet, StudentT, VonMises,
MixtureSameFamily, ImplicitReparam (universal CDF wrapper).

Installation
------------

.. code-block:: bash

    git clone https://github.com/intsystems/implicit-reparameterization-trick.git
    cd implicit-reparameterization-trick
    pip install src/

Quick Start
-----------

Reparameterized sampling from a Beta distribution::

    import torch
    from irt.distributions import Beta

    alpha = torch.tensor([2.0], requires_grad=True)
    beta = torch.tensor([5.0], requires_grad=True)
    dist = Beta(alpha, beta)
    z = dist.rsample(torch.Size([64]))

Wrapping any distribution with a tractable CDF::

    import torch
    from irt.distributions import ImplicitReparam

    loc = torch.tensor(0.0, requires_grad=True)
    base = torch.distributions.Laplace(loc, 1.0)
    dist = ImplicitReparam(base)
    z = dist.rsample(torch.Size([64]))

Mixture of distributions::

    import torch
    from torch.distributions import Categorical
    from irt.distributions import Normal, MixtureSameFamily

    mix_weights = Categorical(torch.tensor([0.3, 0.7]))
    components = Normal(
        torch.tensor([-1.0, 1.0], requires_grad=True),
        torch.tensor([0.5, 0.5]),
    )
    mixture = MixtureSameFamily(mix_weights, components)
    z = mixture.rsample(torch.Size([64]))

References
----------

- `Paper <https://arxiv.org/abs/1805.08498>`_
- `Documentation <https://intsystems.github.io/implicit-reparameterization-trick/>`_
