from autoconduck.model_presets import clean_model_id, curated_model_catalog
from autoconduck.tui.onboarding_models import default_enabled_ids


def test_clean_model_id_cases():
    assert clean_model_id("us/meta-llama/llama-3-3-70b-instruct") == "llama-3-3-70b-instruct"
    assert clean_model_id("us.anthropic.claude-3-5-sonnet-20241022-v2:0") == "claude-3-5-sonnet-20241022-v2:0"
    assert clean_model_id("openai/gpt-4o") == "gpt-4o"
    assert clean_model_id("gpt-4o") == "gpt-4o"


def test_catalog_ids_are_condensed():
    assert all("/" not in row["id"] for row in curated_model_catalog())
    assert all(not row["id"].lower().startswith(("us.", "eu.", "apac.")) for row in curated_model_catalog())


def test_default_enabled_ids():
    small = [{"id": str(i)} for i in range(6)]
    large = [{"id": str(i)} for i in range(7)]
    assert default_enabled_ids(small) == {str(i) for i in range(6)}
    assert default_enabled_ids(large) == set()
    assert default_enabled_ids(large, [{"id": "2"}, {"id": "missing"}]) == {"2"}
