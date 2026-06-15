"""Dataset containers and simple file loading helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from transporteq_discovery.compat import dataclass, field
from transporteq_discovery.config import DatasetConfig

FloatArray = NDArray[np.float64]


@dataclass(slots=True)
class TransportDataset:
    """Transport dataset for either breakthrough or spatiotemporal measurements."""

    time: FloatArray
    concentration: FloatArray
    position: FloatArray | None = None
    inlet: FloatArray | None = None
    name: str = "transport-dataset"
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.time = np.asarray(self.time, dtype=float).reshape(-1)
        self.concentration = np.asarray(self.concentration, dtype=float)
        if self.position is not None:
            self.position = np.asarray(self.position, dtype=float).reshape(-1)
        if self.inlet is not None:
            self.inlet = np.asarray(self.inlet, dtype=float).reshape(-1)

        if self.concentration.ndim not in (1, 2):
            raise ValueError("concentration must be a 1D breakthrough curve or a 2D spatiotemporal array")
        if self.concentration.shape[0] != self.time.shape[0]:
            raise ValueError("time axis must match the first dimension of concentration")
        if self.position is not None and self.concentration.ndim != 2:
            raise ValueError("position can only be provided for 2D spatiotemporal concentration data")
        if self.position is not None and self.concentration.shape[1] != self.position.shape[0]:
            raise ValueError("position axis must match the second dimension of concentration")

    @property
    def is_breakthrough(self) -> bool:
        return self.position is None or self.concentration.ndim == 1 or self.concentration.shape[1] == 1

    @property
    def outlet_signal(self) -> FloatArray:
        if self.concentration.ndim == 1:
            return self.concentration.reshape(-1)
        return self.concentration[:, -1].reshape(-1)

    @classmethod
    def from_npz(cls, config: DatasetConfig) -> "TransportDataset":
        if config.path is None:
            raise ValueError("dataset.path must be provided to load data from disk")
        with np.load(Path(config.path), allow_pickle=False) as data:
            position = data[config.position_key] if config.position_key in data.files else None
            inlet = data[config.inlet_key] if config.inlet_key and config.inlet_key in data.files else None
            return cls(
                time=data[config.time_key],
                concentration=data[config.concentration_key],
                position=position,
                inlet=inlet,
                name=config.name,
            )
