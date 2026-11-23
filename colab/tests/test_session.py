import json

from colab import session


def _manifest(test_extra=()):
    return {"classes": ["A", "B"], "train": ["data/raw/A/1.jpg", "data/raw/B/1.jpg"],
            "validation": ["data/raw/A/2.jpg"], "test": ["data/raw/B/2.jpg", *test_extra]}


def test_read_version(tmp_path):
    path = tmp_path / "VERSION.txt"
    path.write_text("version=abc1234-20261123120000\ncommit=abc\ndirty=no\nsynced_at=2026-11-23T05:00:00Z\n")
    info = session.read_version(path)
    assert info["version"] == "abc1234-20261123120000"
    assert info["dirty"] == "no"


def test_compare_manifests_same():
    result = session.compare_manifests(_manifest(), _manifest())
    assert result["same"] is True
    assert result["counts_a"] == {"train": 2, "validation": 1, "test": 1}


def test_compare_manifests_diff():
    result = session.compare_manifests(_manifest(), _manifest(test_extra=["data/raw/A/9.jpg"]))
    assert result["same"] is False
    assert result["only_in_b"]["test"] == 1
    assert result["only_in_a"]["test"] == 0


def test_save_restore_roundtrip(tmp_path):
    repo, data_dir = tmp_path / "repo", tmp_path / "drive_data"
    (repo / "data" / "raw" / "A").mkdir(parents=True)
    (repo / "data" / "raw" / "A" / "x.jpg").write_bytes(b"jpeg")
    (repo / "outputs").mkdir()
    (repo / "outputs" / "clean_split_manifest.json").write_text(json.dumps(_manifest()))
    session.save_data(repo, data_dir)
    assert (data_dir / "raw.zip").is_file()
    assert (data_dir / "clean_split_manifest.json").is_file()

    fresh = tmp_path / "fresh"
    assert session.restore_data(fresh, data_dir) is True
    assert (fresh / "data" / "raw" / "A" / "x.jpg").read_bytes() == b"jpeg"
    assert json.loads((fresh / "outputs" / "clean_split_manifest.json").read_text()) == _manifest()


def test_restore_without_data_returns_false(tmp_path):
    assert session.restore_data(tmp_path / "repo", tmp_path / "empty") is False


def _code_tree(root):
    for rel, text in {"src/train.py": "a", "colab/session.py": "b", "colab/sync_to_drive.sh": "c", "app.py": "d",
                      "README.md": "not code", "colab/results/E1/metrics.json": "{}",
                      "src/__pycache__/train.cpython-312.pyc": "x"}.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)


def test_code_hash_tracks_code_files_only(tmp_path):
    _code_tree(tmp_path)
    original = session.code_hash(tmp_path)
    (tmp_path / "README.md").write_text("changed docs")
    (tmp_path / "colab/results/E1/metrics.json").write_text('{"a": 1}')
    (tmp_path / "src/__pycache__/train.cpython-312.pyc").write_text("y")
    assert session.code_hash(tmp_path) == original
    (tmp_path / "src/train.py").write_text("a2")
    assert session.code_hash(tmp_path) != original
