from socf import run_self_optimizing_model

def test_import_only():
    # simple smoke test: function exists and is callable
    assert callable(run_self_optimizing_model)
