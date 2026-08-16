from .model_presets import *
from .presets_data import *
from .presets_fallback import *
from .presets_ingest import *
from . import model_presets, presets_data, presets_fallback, presets_ingest
import autoconduck.presets.model_presets as _presets_mod
import autoconduck.presets.presets_data as _data_mod
import autoconduck.presets.presets_fallback as _fb_mod
import autoconduck.presets.presets_ingest as _ingest_mod

for mod in (_presets_mod, _data_mod, _fb_mod, _ingest_mod):
    for k, v in mod.__dict__.items():
        if not k.startswith("__"):
            globals()[k] = v
