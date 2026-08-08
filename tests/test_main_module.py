import importlib.util


def test_main_module_importable():
    assert importlib.util.find_spec("autoconduck.__main__") is not None
