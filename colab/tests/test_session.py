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


def _raises(fn, error):
    try:
        fn()
    except error as exc:
        return exc
    raise AssertionError(f"{fn} did not raise {error.__name__}")


def _mirror(home, account):
    path = home / "Library" / "CloudStorage" / f"GoogleDrive-{account}" / "My Drive" / "DL" / "vnfood_colab"
    path.mkdir(parents=True)
    return path


def test_drive_root_colab(tmp_path):
    assert session.drive_root(True, env={}, home=tmp_path) == session.DRIVE_ROOT


def test_drive_root_env_override(tmp_path):
    env = {"VNFOOD_DRIVE_ROOT": str(tmp_path / "custom" / "vnfood_colab")}
    assert session.drive_root(False, env=env, home=tmp_path) == tmp_path / "custom" / "vnfood_colab"


def test_drive_root_finds_mirror(tmp_path):
    mirror = _mirror(tmp_path, "a@x")
    assert session.drive_root(False, env={}, home=tmp_path) == mirror


def test_drive_root_requires_single_mirror(tmp_path):
    error = _raises(lambda: session.drive_root(False, env={}, home=tmp_path), RuntimeError)
    assert "VNFOOD_DRIVE_ROOT" in str(error)
    _mirror(tmp_path, "a@x"), _mirror(tmp_path, "b@x")
    error = _raises(lambda: session.drive_root(False, env={}, home=tmp_path), RuntimeError)
    assert "VNFOOD_DRIVE_ROOT" in str(error)


def test_choose_device_order():
    assert session.choose_device(True, True) == "cuda"
    assert session.choose_device(False, True) == "mps"
    assert session.choose_device(False, False) == "cpu"


def _saved_data(tmp_path):
    repo, data_dir = tmp_path / "repo", tmp_path / "drive_data"
    (repo / "data" / "raw" / "A").mkdir(parents=True)
    (repo / "data" / "raw" / "A" / "x.jpg").write_bytes(b"jpeg")
    (repo / "outputs").mkdir()
    (repo / "outputs" / "clean_split_manifest.json").write_text(json.dumps(_manifest()))
    session.save_data(repo, data_dir)
    return data_dir


def test_restore_local_manifest_leaves_outputs_alone(tmp_path):
    data_dir, fresh = _saved_data(tmp_path), tmp_path / "fresh"
    target = fresh / "data" / session.MANIFEST_NAME
    assert session.restore_data(fresh, data_dir, manifest_dst=target) is True
    assert json.loads(target.read_text()) == _manifest()
    assert not (fresh / "outputs").exists()


def test_restore_recovers_from_interrupted_extract(tmp_path):
    data_dir, fresh = _saved_data(tmp_path), tmp_path / "fresh"
    (fresh / "data" / ".extract" / "raw").mkdir(parents=True)
    (fresh / "data" / ".extract" / "raw" / "junk.jpg").write_bytes(b"junk")
    assert session.restore_data(fresh, data_dir) is True
    assert (fresh / "data" / "raw" / "A" / "x.jpg").read_bytes() == b"jpeg"
    assert not (fresh / "data" / "raw" / "junk.jpg").exists()
    assert not (fresh / "data" / ".extract").exists()


def test_start_local_paths(tmp_path):
    import os
    repo, root = tmp_path / "repo", tmp_path / "drive" / "vnfood_colab"
    repo.mkdir()
    saved_cwd, saved_env = os.getcwd(), os.environ.get("VNFOOD_DRIVE_ROOT")
    os.environ["VNFOOD_DRIVE_ROOT"] = str(root)
    try:
        env = session.start(repo, on_colab=False)
    finally:
        os.chdir(saved_cwd)
        if saved_env is None:
            os.environ.pop("VNFOOD_DRIVE_ROOT")
        else:
            os.environ["VNFOOD_DRIVE_ROOT"] = saved_env
    assert env["on_colab"] is False and env["repo"] == repo
    assert env["runs"] == root / "runs" and env["data"] == root / "data"
    assert env["manifest"] == repo / "data" / session.MANIFEST_NAME
    assert env["device"] in ("cuda", "mps", "cpu")


def test_notebook_shell_calls_are_safe():
    import re
    sources = sorted((session.REPO / "colab" / "notebooks" / "src").glob("0[123]_*.py"))
    assert len(sources) == 3
    for source in sources:
        for line in source.read_text(encoding="utf-8").splitlines():
            if not line.startswith("!"):
                continue
            assert not line.startswith("!python "), (source.name, line)
            if "colab.train_arch" in line:
                assert '"{sys.executable}"' in line, (source.name, line)
            for name in re.findall(r"\{(RUNS|MANIFEST|REPO)\}", line):
                assert f'"{{{name}}}"' in line, (source.name, line)


def _cell0(source):
    text = source.read_text(encoding="utf-8")
    start = text.index("ON_COLAB = ")
    return text[start:text.index("RUNS, MANIFEST, DEVICE", start)]


def test_cell0_identical_in_notebooks():
    sources = sorted((session.REPO / "colab" / "notebooks" / "src").glob("0[123]_*.py"))
    cells = {_cell0(source) for source in sources}
    assert len(cells) == 1, "cell 0 differs between notebooks"
    assert 'REPO = Path("/content/vnfood")' in cells.pop()
