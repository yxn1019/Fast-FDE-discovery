"""Mittag-Leffler adapter backed by pymittagleffler when available."""

from __future__ import annotations

import tarfile
import types
from functools import lru_cache
from pathlib import Path

import numpy as np

DEFAULT_PYMITTAGLEFFLER_TARBALL = Path(
    r"D:\OneDrive - HHU\ML codes\DL-PDE\Linear_fractional\src\transporteq_discovery\pymittagleffler-0.2.0.tar.gz"
)
FALLBACK_MEMBER = "pymittagleffler-0.2.0/src/pymittagleffler/fallback.py"


@lru_cache(maxsize=1)
def _load_installed_mittag_leffler():
    try:
        from pymittagleffler import mittag_leffler
    except Exception:
        return None
    return mittag_leffler


@lru_cache(maxsize=4)
def _load_tarball_fallback(tarball: Path):
    if not tarball.exists():
        raise ImportError(f"pymittagleffler tarball not found: {tarball}")
    with tarfile.open(tarball) as archive:
        source = archive.extractfile(FALLBACK_MEMBER)
        if source is None:
            raise ImportError(f"{FALLBACK_MEMBER} not found in {tarball}")
        code = source.read().decode("utf-8")

    module = types.ModuleType("_pymittagleffler_fallback")
    exec(compile(code, FALLBACK_MEMBER, "exec"), module.__dict__)
    return module.mittag_leffler_garrappa


def mittag_leffler(
    z: np.ndarray,
    alpha: float,
    beta: float = 1.0,
    *,
    tarball: str | Path = DEFAULT_PYMITTAGLEFFLER_TARBALL,
) -> np.ndarray:
    """Evaluate ``E_{alpha,beta}(z)`` using pymittagleffler.

    This adapter first uses an installed ``pymittagleffler`` package when
    possible. The source-tarball fallback is kept for older environments such
    as the previous Python 3.9 ``sr`` environment.
    """

    values = np.asarray(z, dtype=np.complex128)
    installed = _load_installed_mittag_leffler()
    if installed is not None:
        return np.asarray(installed(values, alpha, beta), dtype=np.complex128)

    fallback = _load_tarball_fallback(Path(tarball))
    return np.asarray(fallback(values, alpha, beta), dtype=np.complex128)
