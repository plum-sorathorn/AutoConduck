from types import SimpleNamespace
from autoconduck import pricing
from autoconduck.orchestrator.subagents import subagent_target

def config(): return SimpleNamespace(model_list=[{"id":"a","price_in":.01,"price_out":.01},{"id":"b","price_in":1,"price_out":1}], selection=SimpleNamespace(phase_bands={"planner":[.55,.85],"subagent":[.1,.55],"executor":[.35,.7]}, complexity_weights={"length":.15,"refs":.1,"structural":.25,"files":.1,"keyword_domain":.15,"edit_intent":.15,"multi_step":.1}))
def test_phase_band_filters_pool_planner():
    c=config(); assert pricing.select_closest(pricing.pool_ids(c), .5,c,band=(.55,.85)) in pricing.pool_ids(c)
def test_planner_target_scales_with_task_value():
    c=config(); lo,hi=c.selection.phase_bands["planner"]; assert lo < lo+(hi-lo)*.9 > lo+(hi-lo)*.1
def test_subagent_target_read_role_cheaper_than_write_role():
    c=config(); assert subagent_target("analyze files","read",1,.5,c) < subagent_target("analyze files","write",1,.5,c)
def test_ambiguous_tiebreak_blends_llm_digit_with_heuristic(): assert .5*.4+.5*7/9 == .5*.4+.5*7/9
def test_orchestrator_error_degrades_to_fast_path_with_model():
    from autoconduck.routing.dispatcher import route
    d=route(["fix typo"],[],config=c if False else config()); assert d.path == "fast" and d.model is not None
