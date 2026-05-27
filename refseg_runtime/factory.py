from importlib import import_module
from typing import Any, Callable


def load_factory(spec: str) -> Callable[..., Any]:
    if ":" not in spec:
        raise ValueError(f"Factory spec must be module:function, got: {spec}")
    module_name, func_name = spec.split(":", 1)
    module = import_module(module_name)
    func = getattr(module, func_name, None)
    if not callable(func):
        raise AttributeError(f"Factory function not found: {spec}")
    return func
