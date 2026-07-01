def test_team_project_package_imports():
    import team_project
    from team_project.data.preprocessing import preprocessing
    from team_project.training.pytorch_loops import pytorch_loops

    assert team_project is not None
    assert preprocessing is not None
    assert pytorch_loops is not None
