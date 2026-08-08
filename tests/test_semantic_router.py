import autoconduck.semantic_router as module


def test_seed_examples_return_valid_matches():
    for text, expected in (("fix this typo", "fast_path"), ("refactor the application", "slow_path")):
        match = module.route(text)
        assert match.route == expected
        assert 0 <= match.confidence <= 1


def test_fallback_scorer_is_safe_without_aurelio(monkeypatch):
    router = module.SemanticRouter()
    router._layer = None
    match = router.route("rename this function")
    assert match.route in {"fast_path", "slow_path"}
    assert 0 <= match.confidence <= 1
