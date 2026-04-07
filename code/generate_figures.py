"""
Generate publication-quality figures and results table for README.
Trains VAEs on dynamically binarized MNIST with multiple seeds and reports mean +/- std.
"""

import math
import sys

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

matplotlib.rcParams["font.family"] = "serif"
matplotlib.rcParams["font.size"] = 10
matplotlib.rcParams["axes.linewidth"] = 0.8
matplotlib.rcParams["figure.dpi"] = 150

sys.path.append("../src")
from irt.distributions import Beta, Gamma, Normal, VonMises

# ---- Data ----


def dynamic_binarize(x):
    return torch.bernoulli(x)


def get_loaders(batch_size=128):
    transform = transforms.ToTensor()
    train = datasets.MNIST("../data", train=True, download=True, transform=transform)
    test = datasets.MNIST("../data", train=False, download=True, transform=transform)
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    return train_loader, test_loader


# ---- Architecture ----


def clamp_positive(x):
    return F.softplus(x).clamp(1e-3, 1e3)


class Encoder(nn.Module):
    def __init__(self, D, n_params=2):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(784, 256), nn.ReLU(), nn.Linear(256, 128), nn.ReLU())
        self.heads = nn.ModuleList([nn.Linear(128, D) for _ in range(n_params)])

    def forward(self, x):
        h = self.net(x.view(x.size(0), -1))
        return [head(h) for head in self.heads]


class Decoder(nn.Module):
    def __init__(self, D, input_dim=None):
        super().__init__()
        d = input_dim or D
        self.net = nn.Sequential(nn.Linear(d, 128), nn.ReLU(), nn.Linear(128, 256), nn.ReLU(), nn.Linear(256, 784))

    def forward(self, z):
        return self.net(z)


# ---- KL divergences ----


def kl_normal(mu, log_sigma):
    return 0.5 * (mu.pow(2) + log_sigma.exp().pow(2) - 1 - 2 * log_sigma).sum(-1)


def kl_gamma(a, b, a0, b0):
    return (
        (a - a0) * torch.digamma(a) - torch.lgamma(a) + torch.lgamma(a0) + a0 * (b.log() - b0.log()) + a * (b0 - b) / b
    ).sum(-1)


def kl_beta(a, b, a0, b0):
    return (
        torch.lgamma(a0)
        + torch.lgamma(b0)
        - torch.lgamma(a0 + b0)
        - torch.lgamma(a)
        - torch.lgamma(b)
        + torch.lgamma(a + b)
        + (a - a0) * torch.digamma(a)
        + (b - b0) * torch.digamma(b)
        + (a0 - a + b0 - b) * torch.digamma(a + b)
    ).sum(-1)


# ---- VAE models ----


class NormalVAE(nn.Module):
    name = "Normal"

    def __init__(self, D):
        super().__init__()
        self.encoder, self.decoder, self.D = Encoder(D), Decoder(D), D

    def forward(self, x, w=1.0):
        mu, ls = self.encoder(x)
        z = Normal(mu, ls.exp().clamp(1e-3, 1e3)).rsample()
        r = F.binary_cross_entropy_with_logits(self.decoder(z), x.view(-1, 784), reduction="none").sum(-1)
        return (r + w * kl_normal(mu, ls)).mean(), r.mean(), kl_normal(mu, ls).mean()

    def sample(self, n, dev):
        return torch.sigmoid(self.decoder(torch.randn(n, self.D, device=dev))).view(-1, 1, 28, 28)

    def encode(self, x):
        return self.encoder(x)[0]


class GammaVAE(nn.Module):
    name = "Gamma(0.3,0.3)"

    def __init__(self, D, a0=0.3, b0=0.3):
        super().__init__()
        self.encoder, self.decoder, self.D, self.a0, self.b0 = Encoder(D), Decoder(D), D, a0, b0

    def forward(self, x, w=1.0):
        ra, rb = self.encoder(x)
        a, b = clamp_positive(ra), clamp_positive(rb)
        z = Gamma(a, b).rsample()
        r = F.binary_cross_entropy_with_logits(self.decoder(z), x.view(-1, 784), reduction="none").sum(-1)
        return (
            (
                r + w * kl_gamma(a, b, torch.tensor(self.a0, device=x.device), torch.tensor(self.b0, device=x.device))
            ).mean(),
            r.mean(),
            kl_gamma(a, b, torch.tensor(self.a0, device=x.device), torch.tensor(self.b0, device=x.device)).mean(),
        )

    def sample(self, n, dev):
        z = Gamma(torch.full((n, self.D), self.a0, device=dev), torch.full((n, self.D), self.b0, device=dev)).rsample()
        return torch.sigmoid(self.decoder(z)).view(-1, 1, 28, 28)

    def encode(self, x):
        ra, rb = self.encoder(x)
        return clamp_positive(ra) / clamp_positive(rb)


class BetaVAE(nn.Module):
    name = "Beta(uniform)"

    def __init__(self, D):
        super().__init__()
        self.encoder, self.decoder, self.D = Encoder(D), Decoder(D), D

    def forward(self, x, w=1.0):
        ra, rb = self.encoder(x)
        a, b = clamp_positive(ra), clamp_positive(rb)
        z = Beta(a, b).rsample()
        r = F.binary_cross_entropy_with_logits(self.decoder(z), x.view(-1, 784), reduction="none").sum(-1)
        return (
            (r + w * kl_beta(a, b, torch.ones_like(a), torch.ones_like(b))).mean(),
            r.mean(),
            kl_beta(a, b, torch.ones_like(a), torch.ones_like(b)).mean(),
        )

    def sample(self, n, dev):
        z = Beta(torch.ones(n, self.D, device=dev), torch.ones(n, self.D, device=dev)).rsample()
        return torch.sigmoid(self.decoder(z)).view(-1, 1, 28, 28)

    def encode(self, x):
        ra, rb = self.encoder(x)
        a, b = clamp_positive(ra), clamp_positive(rb)
        return a / (a + b)


class VonMisesVAE(nn.Module):
    name = "VonMises(uniform)"

    def __init__(self, D):
        super().__init__()
        self.encoder, self.decoder, self.D = Encoder(D), Decoder(D, input_dim=2 * D), D

    def forward(self, x, w=1.0):
        mu, rk = self.encoder(x)
        kappa = clamp_positive(rk)
        dist = VonMises(mu, kappa)
        z = dist.rsample()
        zr = torch.cat([torch.cos(z), torch.sin(z)], -1)
        r = F.binary_cross_entropy_with_logits(self.decoder(zr), x.view(-1, 784), reduction="none").sum(-1)
        kl = dist.log_prob(z).sum(-1) + self.D * math.log(2 * math.pi)
        return (r + w * kl).mean(), r.mean(), kl.mean()

    def sample(self, n, dev):
        z = torch.rand(n, self.D, device=dev) * 2 * math.pi - math.pi
        zr = torch.cat([torch.cos(z), torch.sin(z)], -1)
        return torch.sigmoid(self.decoder(zr)).view(-1, 1, 28, 28)

    def encode(self, x):
        return self.encoder(x)[0]


# ---- Training ----


def train_vae(model, train_loader, test_loader, device, epochs=30, lr=1e-3, anneal=30000):
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    step = 0
    for ep in range(1, epochs + 1):
        model.train()
        for x, _ in train_loader:
            x = dynamic_binarize(x).to(device)
            w = min(1.0, step / anneal)
            loss, _, _ = model(x, w)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            step += 1
    model.eval()
    total, n = 0.0, 0
    with torch.no_grad():
        for x, _ in test_loader:
            x = dynamic_binarize(x).to(device)
            loss, _, _ = model(x, 1.0)
            total += loss.item() * x.size(0)
            n += x.size(0)
    return total / n


# ---- Main ----

if __name__ == "__main__":
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {DEVICE}")
    train_loader, test_loader = get_loaders()

    DIMS = [2, 10, 20]
    SEEDS = [0, 1, 2]
    MAKERS = [NormalVAE, GammaVAE, BetaVAE, VonMisesVAE]
    NAMES = [m.name for m in [NormalVAE(2), GammaVAE(2), BetaVAE(2), VonMisesVAE(2)]]

    results = {}  # (name, D) -> list of NLLs
    d2_models = {}  # name -> model (last seed, D=2)

    for D in DIMS:
        for Maker in MAKERS:
            key = (Maker(D).name, D)
            nlls = []
            for seed in SEEDS:
                torch.manual_seed(seed)
                model = Maker(D)
                nll = train_vae(model, train_loader, test_loader, DEVICE)
                nlls.append(nll)
                if D == 2 and seed == SEEDS[-1]:
                    d2_models[model.name] = model
                print(f"  {key[0]:20s} D={D:2d} seed={seed} NLL={nll:.1f}")
            results[key] = nlls

    # ---- Results table ----
    print("\n" + "=" * 70)
    print("RESULTS TABLE")
    print("=" * 70)

    prior_labels = {
        "Normal": "N(0,1)",
        "Gamma(0.3,0.3)": "Gamma(0.3, 0.3)",
        "Beta(uniform)": "Uniform(0,1)",
        "VonMises(uniform)": "Uniform(-pi, pi)",
    }
    post_labels = {
        "Normal": "N(mu, sigma^2)",
        "Gamma(0.3,0.3)": "Gamma(alpha, beta)",
        "Beta(uniform)": "Beta(alpha, beta)",
        "VonMises(uniform)": "VonMises(mu, kappa)",
    }

    header = f"{'Prior':<20s} {'Posterior':<22s}"
    for D in DIMS:
        header += f"  {'D=' + str(D):>14s}"
    print(header)
    print("-" * len(header))

    table_rows = []
    for name in NAMES:
        row = f"{prior_labels[name]:<20s} {post_labels[name]:<22s}"
        row_data = {"Prior": prior_labels[name], "Posterior": post_labels[name]}
        for D in DIMS:
            nlls = results[(name, D)]
            mean, std = np.mean(nlls), np.std(nlls)
            cell = f"{mean:.1f} +/- {std:.1f}"
            row += f"  {cell:>14s}"
            row_data[f"D={D}"] = (mean, std)
        print(row)
        table_rows.append(row_data)

    # ---- Save table as PNG ----
    fig, ax = plt.subplots(figsize=(9, 2.8))
    ax.axis("off")
    col_labels = ["Prior", "Posterior"] + [f"D={D}" for D in DIMS]
    cell_text = []
    for row in table_rows:
        cells = [row["Prior"], row["Posterior"]]
        for D in DIMS:
            m, s = row[f"D={D}"]
            cells.append(f"{m:.1f} \u00b1 {s:.1f}")
        cell_text.append(cells)

    table = ax.table(cellText=cell_text, colLabels=col_labels, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.0, 1.8)
    for key, cell in table.get_celld().items():
        cell.set_edgecolor("#999999")
        cell.set_linewidth(0.5)
        if key[0] == 0:
            cell.set_facecolor("#e8e8e8")
            cell.set_text_props(weight="bold", fontsize=10)
        else:
            cell.set_facecolor("white")
    fig.savefig("../images/results_table.png", dpi=200, bbox_inches="tight", facecolor="white", pad_inches=0.1)
    plt.close(fig)
    print("\nSaved images/results_table.png")

    # ---- Latent space visualization ----
    fig, axes = plt.subplots(1, 4, figsize=(16, 3.8))
    cmap = plt.cm.tab10
    order = ["Normal", "Gamma(0.3,0.3)", "Beta(uniform)", "VonMises(uniform)"]
    subtitles = ["Normal", "Gamma(0.3, 0.3)", "Beta (uniform prior)", "VonMises (uniform prior)"]

    labels_all = None
    with torch.no_grad():
        for ax, name, subtitle in zip(axes, order, subtitles):
            model = d2_models[name]
            model.eval()
            zs, ys = [], []
            for x, y in test_loader:
                x = x.to(DEVICE)
                z = model.encode(x)
                zs.append(z.cpu())
                ys.append(y)
            z_np = torch.cat(zs, 0).numpy()
            if labels_all is None:
                labels_all = torch.cat(ys, 0).numpy()
            sc = ax.scatter(
                z_np[:, 0], z_np[:, 1], c=labels_all, cmap=cmap, s=0.8, alpha=0.5, vmin=0, vmax=9, rasterized=True
            )
            ax.set_title(subtitle, fontsize=11, pad=6)
            ax.set_aspect("equal")
            ax.tick_params(labelsize=7)
            for spine in ax.spines.values():
                spine.set_linewidth(0.5)

    cbar = fig.colorbar(sc, ax=axes, ticks=range(10), shrink=0.85, aspect=30, pad=0.02)
    cbar.set_label("Digit", fontsize=10)
    cbar.ax.tick_params(labelsize=8)
    fig.tight_layout(rect=[0, 0, 0.95, 1])
    fig.savefig("../images/latent_spaces.png", dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("Saved images/latent_spaces.png")

    # ---- Generated samples grid ----
    n_rows, n_cols = 5, 10
    n = n_rows * n_cols
    fig, axes = plt.subplots(4, 1, figsize=(n_cols * 0.55, 4 * n_rows * 0.55 + 1.5))

    with torch.no_grad():
        for ax, name, subtitle in zip(axes, order, subtitles):
            model = d2_models[name]
            model.eval()
            imgs = model.sample(n, DEVICE).cpu()
            grid = torch.zeros(n_rows * 28, n_cols * 28)
            for i in range(n_rows):
                for j in range(n_cols):
                    grid[i * 28 : (i + 1) * 28, j * 28 : (j + 1) * 28] = imgs[i * n_cols + j, 0]
            ax.imshow(grid.numpy(), cmap="gray", vmin=0, vmax=1, interpolation="nearest")
            ax.set_title(subtitle, fontsize=10, pad=4)
            ax.axis("off")

    fig.tight_layout(h_pad=1.0)
    fig.savefig("../images/generated_samples.png", dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("Saved images/generated_samples.png")
    print("\nDone.")
