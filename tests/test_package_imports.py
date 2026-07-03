def test_team_project_package_imports():
    import team_project
    from team_project.training.pytorch_loops import make_loader, train_once

    assert team_project is not None
    assert make_loader is not None
    assert train_once is not None
