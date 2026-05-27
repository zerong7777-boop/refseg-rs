from __future__ import annotations

import importlib
import os
import sys


ENV_RUNTIME_SITE_PACKAGES = "REFSEG_RUNTIME_SITE_PACKAGES"


def prepend_site_packages(site_packages: str) -> str:
    if site_packages and site_packages not in sys.path:
        sys.path.insert(0, site_packages)
    return site_packages


def _purge_modules(prefixes: tuple[str, ...]) -> None:
    for name in list(sys.modules):
        if any(name == prefix or name.startswith(prefix + ".") for prefix in prefixes):
            sys.modules.pop(name, None)


def _probe_transformers_torch_backend() -> None:
    transformers = importlib.import_module("transformers")
    bert_config_cls = getattr(transformers, "BertConfig")
    getattr(transformers, "AutoTokenizer")
    bert_model_cls = getattr(transformers, "BertModel")
    bert_model_cls(bert_config_cls())


def ensure_transformers_backend(site_packages: str = "") -> str:
    if site_packages:
        prepend_site_packages(site_packages)
        return site_packages

    try:
        _probe_transformers_torch_backend()
        return ""
    except Exception as first_exc:
        fallback = os.environ.get(ENV_RUNTIME_SITE_PACKAGES, "").strip()
        if not fallback:
            raise RuntimeError(
                "transformers is available but its PyTorch backend is unusable. "
                f"Set {ENV_RUNTIME_SITE_PACKAGES} to a site-packages directory with a working torch backend, "
                "or pass site_packages explicitly."
            ) from first_exc

        _purge_modules(("transformers", "tokenizers"))
        prepend_site_packages(fallback)
        try:
            _probe_transformers_torch_backend()
        except Exception as second_exc:
            raise RuntimeError(
                "Unable to initialize a usable transformers + PyTorch backend "
                f"from {ENV_RUNTIME_SITE_PACKAGES} or the current environment."
            ) from second_exc
        return fallback
