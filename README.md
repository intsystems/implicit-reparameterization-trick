<div align="center">  
    <h1> Implicit Reparametrization Trick </h1>
</div>

<div align="center">  
    <img src="images/implicit.webp" width="500px" />
</div>

<p align="center">
    <a href="">
        <img alt="Coverage_2" src="https://github.com/intsystems/implicit-reparameterization-trick/actions/workflows/testing.yml/badge.svg" />
    </a>
    <a href="">
        <img alt="Docs" src="https://github.com/intsystems/implicit-reparameterization-trick/actions/workflows/docs.yml/badge.svg" />
    </a>
</p>

<table>
    <tr>
        <td align="left"> <b> Title </b> </td>
        <td> Implicit Reparametrization Trick for BMM </td>
    </tr>
    <tr>
        <td align="left"> <b> Authors </b> </td>
        <td> Matvei Kreinin, Maria Nikitina, Petr Babkin, Iryna Zabarianska </td>
    </tr>
    <tr>
        <td align="left"> <b> Consultant </b> </td>
        <td> Oleg Bakhteev, PhD </td>
    </tr>
</table>

## 💡 Description

This repository implements an educational project for the Bayesian Multimodeling course. It implements algorithms for sampling from various distributions, using the implicit reparameterization trick.

## 🗃 Scope
We plan to implement the following distributions in our library:
- [x] Gaussian normal distribution (*)
- [x] Dirichlet distribution (Beta distributions)(\*)
- [x] Mixture of the same family distributions (**)
- [x] Student's t-distribution (**) (\*)
- [x] VonMises distribution (***)
- [ ] Sampling from an arbitrary factorized distribution (***)

(\*) - this distribution is already implemented in torch using the explicit reparameterization trick, we will implement it for comparison

(\*\*) - this distribution is added as a backup, their inclusion is questionable

(\*\*\*) - this distribution is not very clear in implementation, its inclusion is questionable

## 📚 Stack

We plan to inherit from the torch.distribution.Distribution class, so we need to implement all the methods that are present in that class.

## 👨‍💻 Usage
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
>>> import irt
>>> params = Encoder(inputs)
>>> mix = irt.Mixture([irt.Normal(*params), irt.Dirichlet(*params)])
>>> deviated = mix.rsample()
>>> outputs = Decoder(deviated)
```

## 📬 Links
- [LinkReview](https://github.com/intsystems/implitic-reparametrization-trick/blob/main/linkreview.md)
- [Plan of project](https://github.com/intsystems/implitic-reparametrization-trick/blob/main/planning.md)
- [BlogPost](blogpost/Blog_post_sketch.pdf)
- [Documentation](https://intsystems.github.io/implicit-reparameterization-trick/)
- [Matvei Kreinin](https://github.com/kreininmv), [Maria Nikitina](https://github.com/NikitinaMaria), [Petr Babkin](https://github.com/petr-parker), [Iryna Zabarianska](https://github.com/Akshiira)

