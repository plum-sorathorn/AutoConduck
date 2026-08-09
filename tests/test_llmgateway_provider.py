import json
import types
from autoconduck import model_presets
from autoconduck.messages_api import litellm_params_for
from autoconduck.tui.onboarding import ModelSourceScreen

def test_llmgateway_preset_shape_and_discovery():
 rows=model_presets.PRESETS["llmgateway"]; assert len(rows)==108
 q=next(x for x in rows if x["id"]=="qwen3.7-flash"); assert (q["price_in"],q["price_out"])==(0.03,0.13)
 assert all(x["base_url"]=="https://devpass.llmgateway.io" and x["api_key_env"]=="LLMGATEWAY_API_KEY" for x in rows)
 assert len(model_presets.discover_models(preset_keys=["llmgateway"],use_litellm=False))==108
 checks={x["id"]: x for x in rows}
 assert (checks["qwen3.6-flash"]["price_in"],checks["qwen3.6-flash"]["price_out"]) == (0.17,0.99)
 assert checks["claude-opus-4-1-20250805"]["tier"] == "expensive"
 assert checks["deepseek-v4-flash"]["tier"] == "budget"
 assert checks["kimi-k3"]["tier"] == "expensive"
 assert checks["gpt-5.2-codex"]["tier"] == "balanced"

def test_llmgateway_catalog_and_fallback(monkeypatch):
 monkeypatch.setattr(model_presets,"_catalog_cache",None); monkeypatch.setattr(model_presets,"_ingest_litellm_costs",lambda *a,**k:{})
 r=next(x for x in model_presets.curated_model_catalog() if x["id"]=="qwen3.7-flash"); assert r["provider"]=="llmgateway" and (r["price_in"],r["price_out"])==(0.03,0.13)
 f=json.loads(model_presets.FALLBACK_PATH.read_text()); assert f["muse-spark-1.2"]["price_out"]==4.25
 assert f["qwen3.7-flash"]["price_in"]==0.03
 assert f["grok-4-5"]["price_out"]==6.0
 assert f["qwen3-vl-flash"]["price_out"]==0.21
 assert any(x["id"]=="gpt-oss-120b" and x["price_in"]==0.05 for x in model_presets.curated_model_catalog())

def test_model_list_entry_supplies_gateway_params():
 cfg=types.SimpleNamespace(model_list=[{"id":"llm-gw-model","base_url":"https://devpass.llmgateway.io","api_key_env":"LLMGATEWAY_API_KEY","enabled":True}],custom_models=[])
 assert litellm_params_for("llm-gw-model",cfg)=={"model":"openai/llm-gw-model","api_base":"https://devpass.llmgateway.io/v1","api_key":"LLMGATEWAY_API_KEY"}

def test_onboarding_exposes_llmgateway(): assert "LLM Gateway" in ModelSourceScreen.SOURCES
