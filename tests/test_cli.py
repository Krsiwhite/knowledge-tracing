from ktlab.cli import main


def test_train_rejects_missing_approval_before_loading_engine(tmp_path, capsys):
    batch = tmp_path / "batch.json"
    batch.write_text('{"runs": []}', encoding="utf-8")
    code = main(["train", "--batch", str(batch), "--approval", str(tmp_path / "missing.json")])
    assert code == 2
    assert "pending explicit user approval" in capsys.readouterr().err
