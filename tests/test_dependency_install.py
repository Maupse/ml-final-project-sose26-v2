import importlib


def test_core_dependencies_importable():
    for module_name in [
        "numpy",
        "pandas",
        "pyarrow",
        "sklearn",
        "joblib",
        "torch",
    ]:
        module = importlib.import_module(module_name)
        assert module is not None
