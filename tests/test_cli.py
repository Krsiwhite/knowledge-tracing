from ktlab.cli import main


def test_train_rejects_missing_approval_before_loading_engine(tmp_path, capsys):
    batch = tmp_path / "batch.json"
    batch.write_text('{"runs": []}', encoding="utf-8")
    code = main(["train", "--batch", str(batch), "--approval", str(tmp_path / "missing.json")])
    assert code == 2
    assert "pending explicit user approval" in capsys.readouterr().err


def test_batch_stops_after_budget_exhaustion_without_attempting_next_run(tmp_path, monkeypatch):
    from ktlab import cli, engine
    calls = []
    batch = {"batch_id": "fixture", "stop_on_noncompleted_run": True,
             "runs": [{"run_id": "first", "config": {}}, {"run_id": "second", "config": {}}]}
    monkeypatch.setattr(cli.p, "validate_batch_approval", lambda *a: batch)
    monkeypatch.setattr(cli.p, "project_root", lambda: tmp_path)
    def fake_train(config, run_dir):
        calls.append(run_dir.name)
        return {"metadata": {"status": "budget_exhausted"}}
    monkeypatch.setattr(engine, "train", fake_train)
    assert main(["train", "--batch", "fixture.json", "--approval", "fixture-approval.json"]) == 2
    assert calls == ["first"]
