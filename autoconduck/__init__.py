__version__ = "0.3.2"

import sys
from . import _compat, auth, cli, launcher, presets, server, harnesses, orchestrator, routing, tui
from .routing import dispatcher, pricing, model_pool, slm_planner

agents = harnesses

# Backwards-compatibility aliases in sys.modules
sys.modules["autoconduck._compat"] = _compat
sys.modules["autoconduck.harnesses"] = harnesses
sys.modules["autoconduck.agents"] = harnesses
sys.modules["autoconduck.auth"] = auth
sys.modules["autoconduck.providers"] = auth.providers
sys.modules["autoconduck.cli"] = cli
sys.modules["autoconduck.cli_launch"] = cli.cli_launch
sys.modules["autoconduck.launcher"] = launcher
sys.modules["autoconduck.launcher_procs"] = launcher.launcher_procs
sys.modules["autoconduck.launcher_shims"] = launcher.launcher_shims
sys.modules["autoconduck.model_presets"] = presets.model_presets
sys.modules["autoconduck.presets_data"] = presets.presets_data
sys.modules["autoconduck.presets_fallback"] = presets.presets_fallback
sys.modules["autoconduck.presets_ingest"] = presets.presets_ingest
sys.modules["autoconduck.server"] = server
sys.modules["autoconduck.server_routes"] = server.server_routes
sys.modules["autoconduck.server_streaming"] = server.server_streaming
sys.modules["autoconduck.messages_api"] = server.messages_api
sys.modules["autoconduck.messages_models"] = server.messages_models
sys.modules["autoconduck.messages_sse"] = server.messages_sse
sys.modules["autoconduck.dispatcher"] = dispatcher
sys.modules["autoconduck.pricing"] = pricing
sys.modules["autoconduck.model_pool"] = model_pool
sys.modules["autoconduck.slm_planner"] = slm_planner
