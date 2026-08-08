import os
import httpx
from pydantic import BaseModel, Field
class CustomEndpoint(BaseModel):
    display_name: str; base_url: str; api_key_env: str | None = None; api_key: str | None = None; models: list[str] = Field(default_factory=list)
def discover_models(endpoint):
    try: return [x["id"] for x in httpx.get(endpoint.base_url.rstrip("/") + "/v1/models", timeout=5).json().get("data", []) if "id" in x]
    except Exception: return []
def generate_litellm_config(endpoint, selected_models):
    key = endpoint.api_key or (os.environ.get(endpoint.api_key_env) if endpoint.api_key_env else None)
    return [{"model_name": m, "litellm_params": {"model": f"openai/{m}", "api_base": endpoint.base_url.rstrip("/") + "/v1", "api_key": key or (f"os.environ/{endpoint.api_key_env}" if endpoint.api_key_env else None)}} for m in selected_models]
