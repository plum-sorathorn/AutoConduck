"""Backwards-compatibility module forwarding to autoconduck.presets.model_presets."""
from autoconduck.presets.model_presets import *
from autoconduck.presets.model_presets import _ingest_litellm_costs, _catalog_cache
import autoconduck.presets.model_presets as _mod

for _k, _v in _mod.__dict__.items():
    if not _k.startswith("__"):
        globals()[_k] = _v
