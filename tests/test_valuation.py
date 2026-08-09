from autoconduck.config import Config
from autoconduck.evaluator import complexity_of

def test_complexity_weights_sum_to_one_default():
    assert sum(Config().selection.complexity_weights.values()) == 1.0
def test_complexity_of_deterministic_fixed_prompt():
    assert abs(complexity_of("fix typo") - 0.201) < 1e-12
def test_complexity_of_edit_intent_raises_value():
    assert complexity_of("fix the race condition") > complexity_of("explain what this does")
def test_complexity_of_multistep_markers():
    assert complexity_of("do this then next finally") > complexity_of("do this")
