Implicit Reparametrization Trick
==========

|test| |codecov| |docs|

.. |test| image:: https://github.com/intsystems/ProjectTemplate/workflows/test/badge.svg
    :target: https://github.com/intsystems/ProjectTemplate/tree/master
    :alt: Test status
    
.. |codecov| image:: https://img.shields.io/codecov/c/github/intsystems/ProjectTemplate/master
    :target: https://app.codecov.io/gh/intsystems/ProjectTemplate
    :alt: Test coverage
    
.. |docs| image:: https://github.com/intsystems/ProjectTemplate/workflows/docs/badge.svg
    :target: https://intsystems.github.io/implicit-reparameterization-trick/
    :alt: Docs status

Description
==========

This repository implements an educational project for the Bayesian Multimodeling course. It implements algorithms for sampling from various distributions, using the implicit reparameterization trick.

Scope
==========

We plan to implement the following distributions in our library:

- `Gaussian normal distribution`
- `Dirichlet distribution (Beta distributions)`
- `Sampling from a mixture of distributions`
- `Sampling from the Student's t-distribution`
- `Sampling from an arbitrary factorized distribution`

Stack
==========

We plan to inherit from the torch.distribution.Distribution class, so we need to implement all the methods that are present in that class.

Usage
==========

In this example, we demonstrate the application of our library using a Variational Autoencoder (VAE) model, where the latent layer is modified by a normal distribution.::

    import torch.distributions.implicit as irt
    params = Encoder(inputs)
    gauss = irt.Normal(*params)
    deviated = gauss.rsample()
    outputs = Decoder(deviated)

In this example, we demonstrate the use of a mixture of distributions using our library.::

    import irt
    params = Encoder(inputs)
    mix = irt.Mixture([irt.Normal(*params), irt.Dirichlet(*params)])
    deviated = mix.rsample()
    outputs = Decoder(deviated)

Links
==========

- `LinkReview <https://github.com/intsystems/implitic-reparametrization-trick/blob/main/linkreview.md>`_
- `Plan of project <https://github.com/intsystems/implitic-reparametrization-trick/blob/main/planning.md>`_
- `BlogPost <Blog_post_sketch.pdf>`_
- `Documentation <https://intsystems.github.io/implicit-reparameterization-trick/>`_
