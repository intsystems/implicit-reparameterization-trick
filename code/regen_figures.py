"""Regenerate latent space and generated samples figures with proper LaTeX rendering."""
import math
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

matplotlib.rcParams["font.family"] = "serif"
matplotlib.rcParams["mathtext.fontset"] = "cm"
matplotlib.rcParams["font.size"] = 11
matplotlib.rcParams["axes.linewidth"] = 0.8

sys.path.append("../src")
from irt.distributions import Beta, Gamma, Normal, VonMises

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def dynamic_binarize(x):
    return torch.bernoulli(x)


def clamp_positive(x):
    return F.softplus(x).clamp(1e-3, 1e3)


class Encoder(nn.Module):
    def __init__(self, D, n=2):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(784, 256), nn.ReLU(), nn.Linear(256, 128), nn.ReLU())
        self.heads = nn.ModuleList([nn.Linear(128, D) for _ in range(n)])

    def forward(self, x):
        h = self.net(x.view(x.size(0), -1))
        return [head(h) for head in self.heads]


class Decoder(nn.Module):
    def __init__(self, D, d=None):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d or D, 128), nn.ReLU(), nn.Linear(128, 256), nn.ReLU(), nn.Linear(256, 784))

    def forward(self, z):
        return self.net(z)


def kl_normal(mu, ls):
    return 0.5 * (mu.pow(2) + ls.exp().pow(2) - 1 - 2 * ls).sum(-1)


def kl_gamma(a, b, a0, b0):
    return ((a - a0) * torch.digamma(a) - torch.lgamma(a) + torch.lgamma(a0) + a0 * (b.log() - b0.log()) + a * (b0 - b) / b).sum(-1)


def kl_beta(a, b, a0, b0):
    return (torch.lgamma(a0) + torch.lgamma(b0) - torch.lgamma(a0 + b0) - torch.lgamma(a) - torch.lgamma(b) + torch.lgamma(a + b) + (a - a0) * torch.digamma(a) + (b - b0) * torch.digamma(b) + (a0 - a + b0 - b) * torch.digamma(a + b)).sum(-1)


class NormalVAE(nn.Module):
    name = "Normal"

    def __init__(self, D):
        super().__init__()
        self.enc, self.dec, self.D = Encoder(D), Decoder(D), D

    def forward(self, x, w=1.0):
        mu, ls = self.enc(x)
        z = Normal(mu, ls.exp().clamp(1e-3, 1e3)).rsample()
        r = F.binary_cross_entropy_with_logits(self.dec(z), x.view(-1, 784), reduction="none").sum(-1)
        return (r + w * kl_normal(mu, ls)).mean(), r.mean(), kl_normal(mu, ls).mean()


class GammaVAE(nn.Module):
    name = "Gamma"

    def __init__(self, D):
        super().__init__()
        self.enc, self.dec, self.D = Encoder(D), Decoder(D), D

    def forward(self, x, w=1.0):
        ra, rb = self.enc(x)
        a, b = clamp_positive(ra), clamp_positive(rb)
        z = Gamma(a, b).rsample()
        r = F.binary_cross_entropy_with_logits(self.dec(z), x.view(-1, 784), reduction="none").sum(-1)
        kl = kl_gamma(a, b, torch.tensor(0.3, device=x.device), torch.tensor(0.3, device=x.device))
        return (r + w * kl).mean(), r.mean(), kl.mean()


class BetaVAE(nn.Module):
    name = "Beta"

    def __init__(self, D):
        super().__init__()
        self.enc, self.dec, self.D = Encoder(D), Decoder(D), D

    def forward(self, x, w=1.0):
        ra, rb = self.enc(x)
        a, b = clamp_positive(ra), clamp_positive(rb)
        z = Beta(a, b).rsample()
        r = F.binary_cross_entropy_with_logits(self.dec(z), x.view(-1, 784), reduction="none").sum(-1)
        kl = kl_beta(a, b, torch.ones_like(a), torch.ones_like(b))
        return (r + w * kl).mean(), r.mean(), kl.mean()


class VonMisesVAE(nn.Module):
    name = "VonMises"

    def __init__(self, D):
        super().__init__()
        self.enc, self.dec, self.D = Encoder(D), Decoder(D, d=2 * D), D

    def forward(self, x, w=1.0):
        mu, rk = self.enc(x)
        kappa = clamp_positive(rk)
        dist = VonMises(mu, kappa)
        z = dist.rsample()
        zr = torch.cat([torch.cos(z), torch.sin(z)], -1)
        r = F.binary_cross_entropy_with_logits(self.dec(zr), x.view(-1, 784), reduction="none").sum(-1)
        kl = dist.log_prob(z).sum(-1) + self.D * math.log(2 * math.pi)
        return (r + w * kl).mean(), r.mean(), kl.mean()


def train_vae(model, train_loader, epochs=30, anneal=30000):
    model = model.to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    step = 0
    for ep in range(1, epochs + 1):
        model.train()
        for x, _ in train_loader:
            x = dynamic_binarize(x).to(DEVICE)
            w = min(1.0, step / anneal)
            loss, _, _ = model(x, w)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            step += 1
    model.eval()
    return model


if __name__ == "__main__":
    transform = transforms.ToTensor()
    train_ds = datasets.MNIST("../data", train=True, download=True, transform=transform)
    test_ds = datasets.MNIST("../data", train=False, download=True, transform=transform)
    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=128, shuffle=False, num_workers=2, pin_memory=True)

    NAMES = ["Normal", "Gamma", "Beta", "VonMises"]
    MAKERS = [NormalVAE, GammaVAE, BetaVAE, VonMisesVAE]

    # Train D=2 models
    d2_models = {}
    for Maker in MAKERS:
        torch.manual_seed(1)
        m = Maker(2)
        m = train_vae(m, train_loader)
        d2_models[m.name] = m
        print(f"Trained {m.name} D=2")

    # ===== Latent spaces =====
    latent_titles = [
        r"Normal: $q = \mathcal{N}(\mu,\,\sigma^2)$",
        r"Gamma: $q = \mathrm{Gamma}(\alpha,\,\beta)$",
        r"Beta: $q = \mathrm{Beta}(\alpha,\,\beta)$",
        r"VonMises: $q = \mathrm{VM}(\mu,\,\kappa)$",
    ]

    fig, axes = plt.subplots(1, 4, figsize=(18, 4.2))
    labels_all = None

    with torch.no_grad():
        for ax, name, title in zip(axes, NAMES, latent_titles):
            model = d2_models[name]
            model.eval()
            zs, ys = [], []
            for x, y in test_loader:
                x = x.to(DEVICE)
                z = model.enc(x)[0]
                zs.append(z.cpu())
                ys.append(y)
            z_np = torch.cat(zs, 0).numpy()
            if labels_all is None:
                labels_all = torch.cat(ys, 0).numpy()
            sc = ax.scatter(z_np[:, 0], z_np[:, 1], c=labels_all, cmap=plt.cm.tab10,
                            s=0.6, alpha=0.5, vmin=0, vmax=9, rasterized=True)
            ax.set_title(title, fontsize=12, pad=8)
            ax.set_xlabel(r"$z_1$", fontsize=10)
            ax.set_ylabel(r"$z_2$", fontsize=10)
            ax.tick_params(labelsize=8)
            for sp in ax.spines.values():
                sp.set_linewidth(0.5)

    cbar = fig.colorbar(sc, ax=axes.tolist(), ticks=range(10), shrink=0.88, aspect=25, pad=0.02)
    cbar.set_label("Digit class", fontsize=10)
    cbar.ax.tick_params(labelsize=8)
    fig.suptitle("2D Latent Space Encodings of MNIST Test Set", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig("../images/latent_spaces.png", dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("Saved latent_spaces.png")

    # ===== Generated samples =====
    sample_titles = [
        r"Normal, prior $\mathcal{N}(0,\,I)$",
        r"Gamma, prior $\mathrm{Gamma}(0.3,\,0.3)$",
        r"Beta, prior $\mathrm{Uniform}(0,\,1)$",
        r"VonMises, prior $\mathrm{Uniform}(-\pi,\,\pi)$",
    ]

    n_rows, n_cols = 5, 8
    n = n_rows * n_cols
    fig, axes = plt.subplots(1, 4, figsize=(18, 5.5))

    with torch.no_grad():
        for ax, name, title in zip(axes, NAMES, sample_titles):
            model = d2_models[name]
            model.eval()
            if name == "Normal":
                z = torch.randn(n, 2, device=DEVICE)
                imgs = torch.sigmoid(model.dec(z)).view(-1, 1, 28, 28).cpu()
            elif name == "Gamma":
                z = Gamma(torch.full((n, 2), 0.3, device=DEVICE), torch.full((n, 2), 0.3, device=DEVICE)).rsample()
                imgs = torch.sigmoid(model.dec(z)).view(-1, 1, 28, 28).cpu()
            elif name == "Beta":
                z = Beta(torch.ones(n, 2, device=DEVICE), torch.ones(n, 2, device=DEVICE)).rsample()
                imgs = torch.sigmoid(model.dec(z)).view(-1, 1, 28, 28).cpu()
            else:
                z = torch.rand(n, 2, device=DEVICE) * 2 * math.pi - math.pi
                zr = torch.cat([torch.cos(z), torch.sin(z)], -1)
                imgs = torch.sigmoid(model.dec(zr)).view(-1, 1, 28, 28).cpu()
            # Grid with 1px separator
            pad = 1
            h, w = 28 + pad, 28 + pad
            grid = torch.ones(n_rows * h + pad, n_cols * w + pad)
            for i in range(n_rows):
                for j in range(n_cols):
                    grid[pad + i * h: pad + i * h + 28, pad + j * w: pad + j * w + 28] = 1.0 - imgs[i * n_cols + j, 0]
            ax.imshow(grid.numpy(), cmap="gray", vmin=0, vmax=1, interpolation="nearest")
            ax.set_title(title, fontsize=11, pad=6)
            ax.axis("off")

    fig.suptitle("Generated Samples from D=2 Latent Models", fontsize=14, y=1.01)
    fig.tight_layout()
    fig.savefig("../images/generated_samples.png", dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("Saved generated_samples.png")
    print("Done.")
