__version__ = "0.2.3"

import sys
from . import auth, cli, launcher, presets, server, agents, orchestrator, routing, tui
from .routing import dispatcher, pricing, evaluator, fast_graph, semantic_router, complexity

# Backwards-compatibility aliases in sys.modules
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
sys.modules["autoconduck.evaluator"] = evaluator
sys.modules["autoconduck.fast_graph"] = fast_graph
sys.modules["autoconduck.semantic_router"] = semantic_router
sys.modules["autoconduck.complexity"] = complexity
