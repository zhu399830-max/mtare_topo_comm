import hashlib
import importlib.util
import json
from pathlib import Path
import zipfile

import pytest


SPEC = importlib.util.spec_from_file_location("source_archiver", Path(__file__).resolve().parents[3] / "tools/v3/archive_frozen_sources.py")
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


def fixture(root):
    (root / "src").mkdir(parents=True)
    (root / "configs/v3").mkdir(parents=True)
    source = root / "src/example.py"
    source.write_bytes(b"answer = 42\n")
    spec = root / "configs/v3/synthetic.json"
    spec.write_text(json.dumps({"source_sha256": {"src/example.py": hashlib.sha256(source.read_bytes()).hexdigest()}}))
    return spec, source


def test_roundtrip_existing_and_cross_root_determinism(tmp_path):
    first, source = fixture(tmp_path / "one")
    a = module.archive(first, module.ARCHIVE_ROOT, tmp_path / "one")
    assert a["status"] == "created_verified" and a["source_files"] == 1
    with zipfile.ZipFile(a["path"]) as bundle:
        assert bundle.read("run_spec.json") == first.read_bytes()
        assert bundle.read("sources/src/example.py") == source.read_bytes()
    before = Path(a["path"]).stat().st_mtime_ns
    source.write_text("later changed source")
    b = module.archive(first, module.ARCHIVE_ROOT, tmp_path / "one")
    assert b["status"] == "exists_verified" and b["sha256"] == a["sha256"]
    assert Path(a["path"]).stat().st_mtime_ns == before
    second, _ = fixture(tmp_path / "two")
    c = module.archive(second, module.ARCHIVE_ROOT, tmp_path / "two")
    assert c["sha256"] == a["sha256"]


def test_source_drift_no_output(tmp_path):
    spec, source = fixture(tmp_path)
    source.write_bytes(b"changed")
    with pytest.raises(ValueError, match="source drift"):
        module.archive(spec, module.ARCHIVE_ROOT, tmp_path)
    assert not (tmp_path / module.ARCHIVE_ROOT).exists()


@pytest.mark.parametrize("name", ["results/run/data.json", "../secret.py", "src/../secret.py", "/tmp/secret.py", "src/scan.npz"])
def test_forbidden_source_rejected_before_read(tmp_path, name):
    spec, _ = fixture(tmp_path)
    spec.write_text(json.dumps({"source_sha256": {name: "a" * 64}}))
    with pytest.raises(ValueError):
        module.archive(spec, module.ARCHIVE_ROOT, tmp_path)


@pytest.mark.parametrize("location", ["source", "parent", "output"])
def test_symlink_rejected(tmp_path, location):
    spec, source = fixture(tmp_path)
    if location == "source":
        other = tmp_path / "other.py"
        source.rename(other)
        source.symlink_to(other)
    elif location == "parent":
        (tmp_path / "src").rename(tmp_path / "other")
        (tmp_path / "src").symlink_to(tmp_path / "other", target_is_directory=True)
    else:
        (tmp_path / "destination").mkdir()
        (tmp_path / module.ARCHIVE_ROOT).symlink_to(tmp_path / "destination", target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        module.archive(spec, module.ARCHIVE_ROOT, tmp_path)


def test_corrupt_archive_never_overwritten(tmp_path):
    spec, _ = fixture(tmp_path)
    result = module.archive(spec, module.ARCHIVE_ROOT, tmp_path)
    path = Path(result["path"])
    path.write_bytes(b"broken")
    with pytest.raises(ValueError, match="corrupt"):
        module.archive(spec, module.ARCHIVE_ROOT, tmp_path)
    assert path.read_bytes() == b"broken"


def test_duplicate_json_and_output_escape(tmp_path):
    spec, _ = fixture(tmp_path)
    with pytest.raises(ValueError, match="output"):
        module.archive(spec, "results/archive", tmp_path)
    spec.write_text('{"source_sha256":{},"source_sha256":{}}')
    with pytest.raises(ValueError, match="duplicate"):
        module.archive(spec, module.ARCHIVE_ROOT, tmp_path)
