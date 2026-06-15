"""Shared utilities for DNS Hx-only Laplace prediction plots."""

from __future__ import annotations

import json
import math
import re
from pathlib import Path

import numpy as np


def nil_nodes(t: float, *, a: float = 6.0, ns: int = 20, nd: int = 19) -> tuple[np.ndarray, np.ndarray]:
    if t <= 0:
        raise ValueError("NIL inversion requires t > 0.")
    n = np.arange(1, ns + 1 + nd + 1, dtype=float)
    alpha = a + (n - 1.0) * np.pi * 1j
    beta = -np.exp(a) * ((-1.0) ** n)
    n_tail = np.arange(1, nd + 1, dtype=float)
    log_terms = (
        math.lgamma(nd + 1.0)
        - np.array([math.lgamma(nd + 2.0 - value) for value in n_tail])
        - np.array([math.lgamma(value) for value in n_tail])
    )
    bdif = np.flip(np.cumsum(np.exp(log_terms))) / (2.0**nd)
    beta[ns + 1 : ns + 1 + nd] *= bdif
    beta[0] /= 2.0
    return alpha / float(t), beta / float(t)


def dirac_profile_vectorized(
    x: np.ndarray,
    *,
    alpha: float,
    velocity: float,
    x0: float,
    t: float,
    mass: float = 1.0,
    nil_a: float = 6.0,
    nil_ns: int = 20,
    nil_nd: int = 19,
) -> np.ndarray:
    """Vectorized inverse Laplace profile for theta=-velocity < 0."""

    v = float(velocity)
    if not (0.0 < v and 0.0 < alpha < 1.0):
        return np.full_like(x, np.nan, dtype=float)
    values = np.zeros_like(x, dtype=float)
    mask = x >= float(x0)
    if not np.any(mask):
        return values
    distances = x[mask] - float(x0)
    s, weights = nil_nodes(t, a=nil_a, ns=nil_ns, nd=nil_nd)
    s_alpha = s ** float(alpha)
    prefactor = float(mass) * (s ** (float(alpha) - 1.0)) / v
    kernel = prefactor[:, None] * np.exp(-(s_alpha[:, None] * distances[None, :]) / v)
    values[mask] = np.real(np.sum(weights[:, None] * kernel, axis=0))
    return values


def load_metadata(value: np.ndarray) -> dict:
    try:
        text = str(value.item())
    except Exception:
        text = str(value)
    return json.loads(text)


def parse_discovery(report_path: Path) -> tuple[float, float, str]:
    text = report_path.read_text(encoding="utf-8")
    alpha_match = re.search(r"^alpha:\s*([-+0-9.eE]+)", text, flags=re.MULTILINE)
    theta_match = re.search(r"^\s*Hx:\s*([-+0-9.eE]+)\s*$", text, flags=re.MULTILINE)
    equation_match = re.search(r"^candidate equation:\s*(.+)$", text, flags=re.MULTILINE)
    if alpha_match is None or theta_match is None:
        raise ValueError(f"Could not parse alpha/Hx coefficient from {report_path}.")
    alpha = float(alpha_match.group(1))
    theta = float(theta_match.group(1))
    if theta >= 0.0:
        raise ValueError(f"Expected theta < 0, got {theta}.")
    equation = equation_match.group(1).strip() if equation_match else ""
    return alpha, -theta, equation


def normalize_mass(y: np.ndarray, dx: float) -> tuple[np.ndarray, float]:
    mass = float(np.sum(np.clip(y, 0.0, None)) * dx)
    if mass <= 0.0 or not math.isfinite(mass):
        return np.zeros_like(y, dtype=float), mass
    return np.clip(y, 0.0, None) / mass, mass


def log_mse(model_norm: np.ndarray, dns_norm: np.ndarray, mask: np.ndarray, epsilon: float) -> float:
    if not np.any(mask):
        return float("inf")
    residual = np.log10(np.clip(model_norm[mask], 0.0, None) + epsilon) - np.log10(
        np.clip(dns_norm[mask], 0.0, None) + epsilon
    )
    return float(np.mean(residual**2))


def profile_stats(y: np.ndarray, x: np.ndarray, dx: float, dns_norm: np.ndarray, mask: np.ndarray, epsilon: float) -> dict:
    y_pos = np.clip(y, 0.0, None)
    mass = float(np.sum(y_pos) * dx)
    normed = y_pos / mass if mass > 0.0 else np.zeros_like(y_pos)
    mean = float(np.sum(x * normed * dx)) if mass > 0.0 else float("nan")
    var = float(np.sum(((x - mean) ** 2) * normed * dx)) if mass > 0.0 else float("nan")
    skew = (
        float(np.sum(((x - mean) ** 3) * normed * dx) / (var**1.5))
        if mass > 0.0 and var > 0.0
        else float("nan")
    )
    denom = max(float(np.linalg.norm(dns_norm[mask])), np.finfo(float).eps)
    return {
        "relative_l2_mass_normalized": float(np.linalg.norm(normed[mask] - dns_norm[mask]) / denom)
        if np.any(mask)
        else float("inf"),
        "log_profile_mse": log_mse(normed, dns_norm, mask, epsilon),
        "mean_x": mean,
        "std_x": math.sqrt(var) if var >= 0.0 else float("nan"),
        "skewness": skew,
        "peak_x": float(x[int(np.argmax(normed))]) if normed.size else float("nan"),
        "peak_value": float(np.max(normed)) if normed.size else float("nan"),
    }
