"""
Stress tests for numerical stability and edge cases of implicit reparameterization.
Checks: NaN/Inf in gradients, extreme parameter values, repeated calls, gradient correctness.
"""

import math
import sys
import traceback

import torch

sys.path.append("../src")
from torch.distributions import Categorical, Independent

from irt.distributions import (
    Beta,
    Dirichlet,
    Gamma,
    MixtureSameFamily,
    Normal,
    StudentT,
    VonMises,
)

PASS = 0
FAIL = 0


def check(name, condition, detail=""):
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  [OK] {name}")
    else:
        FAIL += 1
        print(f"  [FAIL] {name} -- {detail}")


def test_gradient_no_nan_inf(dist_name, make_dist, param, num_samples=1000):
    """Check that gradients are finite (no NaN/Inf)."""
    try:
        param.grad = None
        dist = make_dist()
        samples = dist.rsample(torch.Size([num_samples]))
        loss = samples.sum()
        loss.backward()
        grad = param.grad
        check(f"{dist_name}: grad finite", grad is not None and torch.isfinite(grad).all(), f"grad={grad}")
    except Exception as e:
        check(f"{dist_name}: no crash", False, f"Exception: {e}")


def test_rsample_finite(dist_name, make_dist, num_samples=2000):
    """Check that samples are finite."""
    try:
        dist = make_dist()
        samples = dist.rsample(torch.Size([num_samples]))
        check(
            f"{dist_name}: samples finite",
            torch.isfinite(samples).all(),
            f"non-finite count: {(~torch.isfinite(samples)).sum()}",
        )
    except Exception as e:
        check(f"{dist_name}: no crash on rsample", False, f"Exception: {e}")


def test_repeated_rsample(dist_name, make_dist, repeats=5):
    """Check that calling rsample multiple times doesn't corrupt state."""
    try:
        dist = make_dist()
        shapes = []
        for i in range(repeats):
            s = dist.rsample(torch.Size([10]))
            shapes.append(s.shape)
        check(
            f"{dist_name}: repeated rsample consistent shape",
            all(sh == shapes[0] for sh in shapes),
            f"shapes: {shapes}",
        )
    except Exception as e:
        check(f"{dist_name}: repeated rsample no crash", False, f"Exception: {e}")


def test_grad_magnitude_reasonable(dist_name, make_dist, param, num_samples=5000, max_grad=1e6):
    """Check gradient magnitude is reasonable (not exploding)."""
    try:
        param.grad = None
        dist = make_dist()
        samples = dist.rsample(torch.Size([num_samples]))
        loss = samples.mean()
        loss.backward()
        grad = param.grad
        if grad is not None:
            max_val = grad.abs().max().item()
            check(f"{dist_name}: grad magnitude < {max_grad}", max_val < max_grad, f"max |grad| = {max_val}")
        else:
            check(f"{dist_name}: grad exists", False, "grad is None")
    except Exception as e:
        check(f"{dist_name}: grad magnitude no crash", False, f"Exception: {e}")


# ==============================================================
print("=" * 60)
print("NORMAL - edge cases")
print("=" * 60)

# Very small scale
for scale_val in [1e-6, 1e-4, 1e-2, 1.0, 100.0, 1e4]:
    loc = torch.tensor(0.0, requires_grad=True)
    scale = torch.tensor(scale_val, requires_grad=True)
    tag = f"Normal(0, {scale_val})"
    test_rsample_finite(tag, lambda l=loc, s=scale: Normal(l, s))
    test_gradient_no_nan_inf(tag + " loc", lambda l=loc, s=scale: Normal(l, s), loc)
    test_gradient_no_nan_inf(tag + " scale", lambda l=loc, s=scale: Normal(l, s), scale)

# Very large loc
loc = torch.tensor(1e6, requires_grad=True)
scale = torch.tensor(1.0, requires_grad=True)
test_gradient_no_nan_inf("Normal(1e6, 1) loc", lambda: Normal(loc, scale), loc)
test_repeated_rsample("Normal(1e6, 1)", lambda: Normal(loc, scale))

print()
print("=" * 60)
print("GAMMA - edge cases")
print("=" * 60)

# Very small concentration (near 0)
for conc_val in [0.01, 0.1, 0.5, 1.0, 5.0, 50.0, 500.0]:
    conc = torch.tensor(conc_val, requires_grad=True)
    rate = torch.tensor(1.0, requires_grad=True)
    tag = f"Gamma({conc_val}, 1)"
    test_rsample_finite(tag, lambda c=conc, r=rate: Gamma(c, r))
    test_gradient_no_nan_inf(tag + " rate", lambda c=conc, r=rate: Gamma(c, r), rate)

# Very large rate
for rate_val in [0.01, 100.0, 1e4]:
    conc = torch.tensor(2.0, requires_grad=True)
    rate = torch.tensor(rate_val, requires_grad=True)
    tag = f"Gamma(2, {rate_val})"
    test_rsample_finite(tag, lambda c=conc, r=rate: Gamma(c, r))
    test_gradient_no_nan_inf(tag + " rate", lambda c=conc, r=rate: Gamma(c, r), rate)

test_repeated_rsample("Gamma(2,1)", lambda: Gamma(torch.tensor(2.0), torch.tensor(1.0)))

print()
print("=" * 60)
print("BETA - edge cases")
print("=" * 60)

# Symmetric, asymmetric, near-boundary concentrations
test_cases_beta = [
    (0.1, 0.1),  # Very small — U-shaped
    (0.5, 0.5),  # Jeffreys prior
    (1.0, 1.0),  # Uniform
    (2.0, 5.0),  # Asymmetric
    (50.0, 50.0),  # Concentrated at 0.5
    (100.0, 1.0),  # Concentrated near 1
    (0.01, 100.0),  # Extreme asymmetry
]

for a, b in test_cases_beta:
    alpha = torch.tensor(a, requires_grad=True)
    beta_p = torch.tensor(b, requires_grad=True)
    tag = f"Beta({a}, {b})"
    test_rsample_finite(tag, lambda al=alpha, be=beta_p: Beta(al, be))
    test_gradient_no_nan_inf(tag + " alpha", lambda al=alpha, be=beta_p: Beta(al, be), alpha)
    test_gradient_no_nan_inf(tag + " beta", lambda al=alpha, be=beta_p: Beta(al, be), beta_p)

test_repeated_rsample("Beta(2,2)", lambda: Beta(torch.tensor(2.0), torch.tensor(2.0)))

print()
print("=" * 60)
print("DIRICHLET - edge cases")
print("=" * 60)

for conc_vals in [[0.1, 0.1, 0.1], [1.0, 1.0, 1.0], [10.0, 10.0, 10.0], [0.01, 0.01, 100.0], [50.0, 50.0, 50.0]]:
    conc = torch.tensor(conc_vals, requires_grad=True)
    tag = f"Dirichlet({conc_vals})"
    test_rsample_finite(tag, lambda c=conc: Dirichlet(c))
    test_gradient_no_nan_inf(tag, lambda c=conc: Dirichlet(c), conc)

# High-dimensional
conc_high = torch.ones(50, requires_grad=True)
test_rsample_finite("Dirichlet(dim=50)", lambda: Dirichlet(conc_high))
test_gradient_no_nan_inf("Dirichlet(dim=50)", lambda: Dirichlet(conc_high), conc_high)

test_repeated_rsample("Dirichlet([1,1,1])", lambda: Dirichlet(torch.tensor([1.0, 1.0, 1.0])))

print()
print("=" * 60)
print("STUDENT-T - edge cases")
print("=" * 60)

# Various df values (heavy tails to near-normal)
for df_val in [0.5, 1.0, 2.0, 3.0, 5.0, 30.0, 100.0]:
    df = torch.tensor(df_val, requires_grad=True)
    loc = torch.tensor(0.0, requires_grad=True)
    scale = torch.tensor(1.0, requires_grad=True)
    tag = f"StudentT(df={df_val})"
    test_rsample_finite(tag, lambda d=df, l=loc, s=scale: StudentT(d, l, s))
    test_gradient_no_nan_inf(tag + " loc", lambda d=df, l=loc, s=scale: StudentT(d, l, s), loc)
    test_gradient_no_nan_inf(tag + " scale", lambda d=df, l=loc, s=scale: StudentT(d, l, s), scale)

# Large scale
loc = torch.tensor(0.0, requires_grad=True)
scale = torch.tensor(1000.0, requires_grad=True)
df = torch.tensor(5.0, requires_grad=True)
test_rsample_finite("StudentT(5, 0, 1000)", lambda: StudentT(df, loc, scale))
test_gradient_no_nan_inf("StudentT(5, 0, 1000) scale", lambda: StudentT(df, loc, scale), scale)

# State not mutated after multiple calls
df2 = torch.tensor(5.0, requires_grad=True)
loc2 = torch.tensor(1.0, requires_grad=True)
scale2 = torch.tensor(2.0, requires_grad=True)
dist_st = StudentT(df2, loc2, scale2)
for i in range(5):
    _ = dist_st.rsample(torch.Size([10]))
check("StudentT: batch_shape stable after 5 rsamples", dist_st.batch_shape == torch.Size())
check("StudentT: loc unchanged after 5 rsamples", dist_st.loc.shape == torch.Size())
check("StudentT: scale unchanged after 5 rsamples", dist_st.scale.shape == torch.Size())

print()
print("=" * 60)
print("VON MISES - edge cases")
print("=" * 60)

# Various concentrations
for conc_val in [0.001, 0.1, 1.0, 5.0, 10.0, 50.0, 100.0, 500.0]:
    loc = torch.tensor(0.0, requires_grad=True)
    conc = torch.tensor(conc_val, requires_grad=True)
    tag = f"VonMises(0, {conc_val})"
    test_rsample_finite(tag, lambda l=loc, c=conc: VonMises(l, c))
    test_gradient_no_nan_inf(tag + " loc", lambda l=loc, c=conc: VonMises(l, c), loc)
    test_gradient_no_nan_inf(tag + " conc", lambda l=loc, c=conc: VonMises(l, c), conc)

# Loc at boundary of [-pi, pi]
for loc_val in [-math.pi + 0.01, 0.0, math.pi - 0.01]:
    loc = torch.tensor(loc_val, requires_grad=True)
    conc = torch.tensor(5.0, requires_grad=True)
    tag = f"VonMises(loc={loc_val:.2f}, 5)"
    test_rsample_finite(tag, lambda l=loc, c=conc: VonMises(l, c))
    test_gradient_no_nan_inf(tag + " loc", lambda l=loc, c=conc: VonMises(l, c), loc)

test_repeated_rsample("VonMises(0,5)", lambda: VonMises(torch.tensor(0.0), torch.tensor(5.0)))

print()
print("=" * 60)
print("MIXTURE SAME FAMILY - edge cases")
print("=" * 60)

# Balanced vs unbalanced weights
for probs in [[0.5, 0.5], [0.99, 0.01], [0.01, 0.99]]:
    loc = torch.tensor([0.0, 5.0], requires_grad=True)
    scale = torch.tensor([1.0, 1.0], requires_grad=True)
    comp = Normal(loc, scale)
    mix = Categorical(torch.tensor(probs))
    tag = f"MSF(probs={probs})"
    test_rsample_finite(
        tag, lambda l=loc, s=scale, p=probs: MixtureSameFamily(Categorical(torch.tensor(p)), Normal(l, s))
    )
    test_gradient_no_nan_inf(
        tag + " loc", lambda l=loc, s=scale, p=probs: MixtureSameFamily(Categorical(torch.tensor(p)), Normal(l, s)), loc
    )

# Multivariate components
loc_mv = torch.tensor([[0.0, 1.0], [2.0, 3.0]], requires_grad=True)
scale_mv = torch.tensor([[1.0, 1.0], [1.0, 1.0]], requires_grad=True)
tag = "MSF(multivariate)"
test_rsample_finite(
    tag, lambda: MixtureSameFamily(Categorical(torch.tensor([0.5, 0.5])), Independent(Normal(loc_mv, scale_mv), 1))
)
test_gradient_no_nan_inf(
    tag + " loc",
    lambda: MixtureSameFamily(Categorical(torch.tensor([0.5, 0.5])), Independent(Normal(loc_mv, scale_mv), 1)),
    loc_mv,
)

test_repeated_rsample(
    "MSF(Normal)",
    lambda: MixtureSameFamily(
        Categorical(torch.tensor([0.5, 0.5])), Normal(torch.tensor([0.0, 3.0]), torch.tensor([1.0, 1.0]))
    ),
)


print()
print("=" * 60)
print("GRADIENT CORRECTNESS - finite differences comparison")
print("=" * 60)


def compare_fd(dist_name, make_dist_fn, param_val, f_fn, eps=5e-3, n_samples=50000, tol=0.2):
    """Compare implicit reparam gradient with finite difference estimate.

    Args:
        make_dist_fn: callable(tensor_with_grad) -> distribution  (for implicit reparam)
                      also called with plain float for finite differences
        param_val: float, the parameter value
    """
    # Finite differences
    torch.manual_seed(0)
    samples_p = make_dist_fn(param_val + eps).rsample(torch.Size([n_samples]))
    torch.manual_seed(0)
    samples_m = make_dist_fn(param_val - eps).rsample(torch.Size([n_samples]))
    grad_fd = (f_fn(samples_p).mean() - f_fn(samples_m).mean()) / (2 * eps)

    # Implicit reparam
    param = torch.tensor(param_val, requires_grad=True)
    dist = make_dist_fn(param)
    torch.manual_seed(42)
    samples = dist.rsample(torch.Size([n_samples]))
    loss = f_fn(samples).mean()
    loss.backward()
    grad_ir = param.grad

    if grad_ir is None:
        check(f"{dist_name}: grad exists", False, "grad is None")
        return

    diff = abs(grad_ir.item() - grad_fd.item())
    check(
        f"{dist_name}: |IR - FD| < {tol}  (IR={grad_ir.item():.4f}, FD={grad_fd.item():.4f})",
        diff < tol,
        f"diff={diff:.4f}",
    )


# Normal
for loc_val, scale_val in [(0.0, 1.0), (5.0, 0.1), (-3.0, 10.0)]:
    compare_fd(
        f"Normal(loc={loc_val}, scale={scale_val}) d/dloc E[x]",
        lambda v, s=scale_val: Normal(v, s),
        loc_val,
        lambda x: x,
    )

    compare_fd(
        f"Normal(loc={loc_val}, scale={scale_val}) d/dscale E[x^2]",
        lambda v, l=loc_val: Normal(l, v),
        scale_val,
        lambda x: x**2,
    )

# Gamma
for conc_val, rate_val in [(2.0, 1.0), (0.5, 2.0), (10.0, 0.5)]:
    compare_fd(
        f"Gamma({conc_val}, rate={rate_val}) d/drate E[x]", lambda v, c=conc_val: Gamma(c, v), rate_val, lambda x: x
    )

# Beta
for a_val, b_val in [(2.0, 2.0), (5.0, 1.0), (0.5, 0.5)]:
    compare_fd(f"Beta(alpha={a_val}, beta={b_val}) d/dalpha E[x]", lambda v, b=b_val: Beta(v, b), a_val, lambda x: x)

# StudentT
for df_val in [3.0, 10.0, 30.0]:
    compare_fd(f"StudentT(df={df_val}) d/dloc E[x]", lambda v, d=df_val: StudentT(d, v, 1.0), 0.0, lambda x: x)

    compare_fd(
        f"StudentT(df={df_val}) d/dscale E[x^2]", lambda v, d=df_val: StudentT(d, 0.0, v), 1.0, lambda x: x**2, tol=0.5
    )

# VonMises
for conc_val in [2.0, 10.0, 50.0]:
    compare_fd(
        f"VonMises(loc=0.5, conc={conc_val}) d/dloc E[cos(x)]",
        lambda v, c=conc_val: VonMises(v, c),
        0.5,
        lambda x: torch.cos(x),
        tol=0.15,
    )

print()
print("=" * 60)
print(f"RESULTS: {PASS} passed, {FAIL} failed out of {PASS + FAIL} checks")
print("=" * 60)
if FAIL > 0:
    sys.exit(1)
