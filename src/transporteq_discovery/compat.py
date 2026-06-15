"""Small compatibility helpers for supported and legacy Python runtimes."""

from __future__ import annotations

import dataclasses as _dataclasses
import sys
from typing import Any

asdict = _dataclasses.asdict
field = _dataclasses.field


def dataclass(_cls: type | None = None, **kwargs: Any):
    """Wrap ``dataclasses.dataclass`` while tolerating ``slots`` on Python 3.9."""

    if sys.version_info < (3, 10):
        kwargs.pop("slots", None)
    return _dataclasses.dataclass(_cls, **kwargs)
