from __future__ import annotations

from typing import Any, Dict


def build_native_refseg_runtime(**_: Dict[str, Any]) -> object:
    raise NotImplementedError(
        "A full native XeMapRefSeg port is not available yet. "
        "Use this package's dataset/metrics/checkpoint utilities first, "
        "then wire the future native model through a factory."
    )
