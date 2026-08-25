import os
import httpx
from pydantic import BaseModel, Field
from autoconduck.config import normalize_api_base, resolve_api_key
class CustomEndpoint(BaseModel):
    display_name: str; base_url: str; anthropic_base_url: str | None = None; api_key_env: str | None = None; api_key: str | None = None; models: list[str] = Field(default_factory=list)
def discover_models(endpoint: CustomEndpoint | Any) -> list[str]:
    try:
        data = endpoint.model_dump() if hasattr(endpoint, "model_dump") else (endpoint if isinstance(endpoint, dict) else {})
        display_name = getattr(endpoint, "display_name", "") or data.get("display_name", "")
        key = resolve_api_key(data, display_name)
        headers = {"Authorization": f"Bearer {key}"} if key else {}
        base_url = getattr(endpoint, "base_url", "") or data.get("base_url", "")
        if not base_url:
            return []
        url = base_url.rstrip("/") + "/v1/models"
        resp = httpx.get(url, headers=headers, timeout=10)
        if resp.status_code == 200:
            models_data = resp.json().get("data", [])
            return [x["id"] for x in models_data if isinstance(x, dict) and "id" in x]
        return []
    except Exception:
        return []
def generate_litellm_config(endpoint, selected_models):
    key = resolve_api_key(endpoint.model_dump(), endpoint.display_name)
    return [{"model_name": m, "litellm_params": {"model": f"openai/{m}", "api_base": normalize_api_base(endpoint.base_url), "api_key": key or (f"os.environ/{endpoint.api_key_env}" if endpoint.api_key_env else None)}} for m in selected_models]
