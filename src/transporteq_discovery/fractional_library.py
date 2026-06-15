"""Candidate libraries for time-fractional PDE discovery."""

from __future__ import annotations

import numpy as np

from transporteq_discovery.compat import dataclass
from transporteq_discovery.fractional_operator import FractionalDifferentialOperator
from transporteq_discovery.models import LibraryTerm


@dataclass(slots=True)
class FractionalLibraryConfig:
    """Options for building a fractional PDE candidate library."""

    max_spatial_derivative: int = 3
    include_nonlinear: bool = True
    include_remainders: bool = True
    reference_alpha: float | None = None
    eps: float = 1.0e-12


class FractionalLibraryBuilder:
    """Build RHS terms and Taylor-style linearization remainders."""

    def __init__(
        self,
        operator: FractionalDifferentialOperator,
        config: FractionalLibraryConfig | None = None,
    ) -> None:
        self.operator = operator
        self.config = config or FractionalLibraryConfig()

    @property
    def sample_count(self) -> int:
        return int(self.operator.field.size)

    def _flatten(self, values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=float).reshape(-1)
        if values.shape[0] != self.sample_count:
            raise ValueError("term values must flatten to the grid sample count")
        if not np.all(np.isfinite(values)):
            raise FloatingPointError("candidate term contains non-finite values")
        return values

    def base_terms(self) -> list[LibraryTerm]:
        """Return the default integer-order RHS library."""

        u = self.operator.field
        ux = self.operator.space_derivative(1)
        uxx = self.operator.space_derivative(2)
        terms = [
            LibraryTerm("u", self._flatten(u), metadata={"kind": "base"}),
            LibraryTerm("u_x", self._flatten(ux), metadata={"kind": "base"}),
            LibraryTerm("u_xx", self._flatten(uxx), metadata={"kind": "base"}),
        ]
        if self.config.max_spatial_derivative >= 3:
            uxxx = self.operator.space_derivative(3)
            terms.append(LibraryTerm("u_xxx", self._flatten(uxxx), metadata={"kind": "base"}))
        if self.config.include_nonlinear:
            terms.extend(
                [
                    LibraryTerm("u^2", self._flatten(u**2), metadata={"kind": "nonlinear"}),
                    LibraryTerm("u*u_x", self._flatten(u * ux), metadata={"kind": "nonlinear"}),
                    LibraryTerm("u*u_xx", self._flatten(u * uxx), metadata={"kind": "nonlinear"}),
                ]
            )
        return terms

    def time_log_weight(self) -> np.ndarray:
        """Finite time-domain proxy for the Laplace Taylor ``log`` factor."""

        time = self.operator.time
        span = max(float(time[-1] - time[0]), self.config.eps)
        weights = np.log1p((time - time[0]) / span)
        return np.repeat(weights, self.operator.position.size)

    def remainder_terms(self, reference_alpha: float) -> list[LibraryTerm]:
        """Return pre-defined linearization remainder candidates."""

        alpha0 = float(reference_alpha)
        caputo = self._flatten(self.operator.caputo_time_derivative(alpha0))
        log_t = self.time_log_weight()
        log_corr = caputo * log_t
        ux = self._flatten(self.operator.space_derivative(1))
        uxx = self._flatten(self.operator.space_derivative(2))

        return [
            LibraryTerm(
                f"Caputo({alpha0:.4g})*log(t)",
                self._flatten(log_corr),
                mechanism_specific=True,
                metadata={"kind": "linearization_remainder", "power": 1, "alpha0": alpha0},
            ),
            LibraryTerm(
                f"Caputo({alpha0:.4g})*(log(t))^2",
                self._flatten(caputo * log_t**2),
                mechanism_specific=True,
                metadata={"kind": "linearization_remainder", "power": 2, "alpha0": alpha0},
            ),
            LibraryTerm(
                f"Caputo({alpha0:.4g})*(log(t))^3",
                self._flatten(caputo * log_t**3),
                mechanism_specific=True,
                metadata={"kind": "linearization_remainder", "power": 3, "alpha0": alpha0},
            ),
            LibraryTerm(
                "log_corr*u_x",
                self._flatten(log_corr * ux),
                mechanism_specific=True,
                metadata={"kind": "mixed_remainder", "alpha0": alpha0},
            ),
            LibraryTerm(
                "log_corr*u_xx",
                self._flatten(log_corr * uxx),
                mechanism_specific=True,
                metadata={"kind": "mixed_remainder", "alpha0": alpha0},
            ),
        ]

    def all_terms(self, reference_alpha: float | None = None) -> list[LibraryTerm]:
        """Return base terms plus optional remainder candidates."""

        terms = self.base_terms()
        alpha0 = self.config.reference_alpha if reference_alpha is None else reference_alpha
        if self.config.include_remainders and alpha0 is not None:
            terms.extend(self.remainder_terms(alpha0))
        return terms
