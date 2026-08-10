from autoconduck.tui.onboarding_models import search_match


def test_search_match_normalizes_terms_and_fields():
    assert search_match("qwen3.7flash", "qwen3.7-flash")
    assert search_match("gpt5", "gpt-5.6-luna")
    assert search_match("QwEn", "qwen-3")
    assert search_match("gpt_5", "gpt-5.6-luna")
    assert not search_match("claude", "gpt-5.6-luna")
    assert search_match("", "anything")
