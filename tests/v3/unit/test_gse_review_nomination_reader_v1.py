"""Temporary synthetic files only; never open research data."""
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path

import numpy as np
from numcodecs import Blosc
import pytest
import zarr

from mtare_topo.data.gse_review_nomination_reader_v1 import ReviewNominationReader as CardReader, digest, FIELDS
from mtare_topo.data.gse_review_nomination_v1 import nominate_interval
from mtare_topo.data.gse_review_source_index_v1 import VARIANTS
from tests.v3.unit.test_gse_review_nomination_v1 import document, interval, PARENT


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def ReviewNominationReader(root, draft_fixture, seals):
    # The real constructor receives the full new-card scope. The old source
    # draft remains unchanged; codebook paths are an explicit separate grant.
    from copy import deepcopy
    draft = deepcopy(draft_fixture)
    paths = draft.pop("codebook_paths", None)
    return CardReader(root, {"source_scope": draft,
        "actual_read_scope": {"codebook_paths": paths}}, seals)


def reseal(root, scope):
    sources = {}
    for role, folders in (("construction", ("construction", "codebooks")), ("teacher", ("teacher",))):
        path = root / (role + "_seal.txt")
        files = sorted(p for folder in folders for p in (root / folder).rglob("*") if p.is_file())
        path.write_text("".join(sha(p) + "  " + str(p.relative_to(root)) + "\n" for p in files)
            + "f" * 64 + "  forbidden_C08/never_resolve/secret.zarr/.zattrs\n")
        sources[role] = dict(path=path.name, sha256=sha(path))
    scope["task_row_summary_sha256"] = digest(scope["task_row_summary"])
    return sources


def fixture(root):
    source = interval(q=23)
    source = replace(source, variants=tuple(replace(v, sequence_rows=tuple(range(250, 273)),
        frame_rows=tuple(tuple(n + 250 for n in fs) for fs in v.frame_rows)) for v in source.variants))
    inventory = root / "inventory" / (PARENT + "_identity_intervals.json")
    write(inventory, dict(parent_id=PARENT, partition="fit", duration_s=None, intervals=[asdict(source)]))
    scope = dict(eligible_parents=[PARENT], task_row_summary=[], codebook_paths={},
        source_roots=dict(construction="construction", teacher="teacher"),
        inventory_source_files={str(inventory.relative_to(root)): sha(inventory)})
    for v in source.variants:
        doc = document(); doc.update(parent_id=PARENT, geometry_realization=v.variant)
        doc["base_construction"].update(endpoint_attachment_mode="free_space_overlap", node_degree_source="edge_incidence")
        for op in doc["base_construction"]["composition_operations"]:
            op["operation"] = "endpoint_union"
        write(root / "construction/fit" / (v.task + ".json"), doc)
        path = "codebooks/fit/" + v.task + ".json"; scope["codebook_paths"][v.task] = path
        write(root / path, dict(schema_version="primitive_membership_codebook_v1", parent_id=PARENT, geometry_realization=v.variant,
            primitive_ids=[p["primitive_id"] for p in doc["realized_primitives"]], source_sets=[[], [0]]))
        group = zarr.open_group(str(root / "teacher/fit" / (v.task + ".zarr")), mode="w")
        group.attrs.update(schema_version="primitive_relation_p1b_teacher_shard_v1", parent_id=PARENT,
            partition="fit", geometry_realization=v.variant, maximum_slots=32, window_frames=5, student_identity_input_forbidden=True)
        for name, (tail, dtype) in FIELDS.items():
            data = np.zeros((600, *tail), dtype=dtype)
            if name == "primitive_index":
                data[:] = -1; data[:, :4] = np.arange(4)
            elif name == "primitive_mask":
                data[:, :4] = 1
            else:
                matrix = np.zeros((600, 32, 32), dtype=np.uint8)
                matrix[:, 0, 3] = matrix[:, 3, 0] = 1
                data = np.packbits(matrix, axis=2, bitorder="little")
            group.create_dataset(name, data=data, chunks=(256, *tail), compressor=Blosc(cname="zstd", clevel=5, shuffle=2))
        group.create_dataset("range_m", data=np.ones((600, 16, 720)), chunks=(16, 16, 720))
        entries = [dict(row=r, source_sequence_id=s, frame_rows=list(f)) for r, s, f in zip(v.sequence_rows, v.source_sequence_ids, v.frame_rows)]
        scope["task_row_summary"].append(dict(task=v.task, source_partition="fit", split="fit", source_sequence_count=600,
            row_count=23, unique_frame_count=27, unique_rows_sha256=digest(list(v.sequence_rows)), row_identity_sha256=digest(entries)))
    return scope, reseal(root, scope)


def target(root, field, suffix):
    return root / "teacher/fit" / (PARENT + "__" + VARIANTS[0] + ".zarr") / field / suffix


def test_exact_chunk_subset_identity_mapping_and_nomination(tmp_path, monkeypatch):
    scope, seals = fixture(tmp_path)
    original = Path.resolve
    def guarded(path, *args, **kwargs):
        assert "C08" not in str(path) and "range_m" not in str(path)
        return original(path, *args, **kwargs)
    monkeypatch.setattr(Path, "resolve", guarded)
    reader = ReviewNominationReader(tmp_path, scope, seals); output = reader.read_parent(PARENT)
    assert output.codebook_order_verified and output.continuity_requires_source_audit
    assert len(output.intervals) == 1 and len(output.overlap_by_task) == 3
    assert len(reader.chunk_reads) == 18
    assert sum(x["decoded_logical_rows"] for x in reader.chunk_reads) == 4608
    assert sum(x["effective_rows"] for x in reader.chunk_reads) == 207
    assert all(x["chunk"] in (0, 1) for x in reader.chunk_reads)
    assert all("range_m" not in p and "C08" not in p for p in reader.opened)
    result = nominate_interval(parent_id=PARENT, split="fit", interval=output.intervals[0],
        construction=output.construction, overlap_by_variant=output.overlaps_for_interval(output.intervals[0]))
    assert result.counts["eligible_windows"] == 3 and result.training_labels_created == 0
    assert {x.stratum for x in result.nominations} == {"junction", "terminal", "corridor", "alias"}
    assert all(sha(tmp_path / path) == value for path, value in reader.opened.items())


@pytest.mark.parametrize("field,key,value", [
    ("primitive_index", "dtype", "<f4"), ("primitive_mask", "chunks", [128, 32]),
    ("disconnected_overlap_packed", "shape", [600, 32, 8]),
    ("primitive_index", "filters", [{"id": "delta", "dtype": "<i4"}]),
    ("primitive_index", "order", "F"), ("primitive_index", "dimension_separator", "/"),
])
def test_headers_fail_before_any_chunk(tmp_path, field, key, value):
    scope, _ = fixture(tmp_path); path = target(tmp_path, field, ".zarray")
    data = json.loads(path.read_text()); data[key] = value; write(path, data)
    reader = ReviewNominationReader(tmp_path, scope, reseal(tmp_path, scope))
    with pytest.raises(ValueError, match="before payload"): reader.read_parent(PARENT)
    assert reader.chunk_reads == []


@pytest.mark.parametrize("kind", ["missing_file", "missing_seal", "drift", "truncated_valid_hash"])
def test_missing_damaged_chunk_never_zero_filled(tmp_path, kind):
    scope, seals = fixture(tmp_path); path = target(tmp_path, "primitive_index", "0.0")
    if kind in ("missing_file", "missing_seal"):
        path.unlink()
    else:
        path.write_bytes(path.read_bytes()[:12])
    if kind in ("missing_seal", "truncated_valid_hash"):
        seals = reseal(tmp_path, scope)
    reader = ReviewNominationReader(tmp_path, scope, seals)
    with pytest.raises(ValueError): reader.read_parent(PARENT)


@pytest.mark.parametrize("kind", ["construction", "codebook", "header", "inventory"])
def test_duplicate_json_keys_rejected(tmp_path, kind):
    scope, seals = fixture(tmp_path)
    path = {"construction": tmp_path / "construction/fit" / (PARENT + "__" + VARIANTS[0] + ".json"),
        "codebook": tmp_path / scope["codebook_paths"][PARENT + "__" + VARIANTS[0]],
        "header": target(tmp_path, "primitive_index", ".zarray"),
        "inventory": tmp_path / next(iter(scope["inventory_source_files"]))}[kind]
    text = path.read_text(); path.write_text('{"duplicate":0,"duplicate":1,' + text[1:])
    if kind == "inventory": scope["inventory_source_files"][str(path.relative_to(tmp_path))] = sha(path)
    reader = ReviewNominationReader(tmp_path, scope, reseal(tmp_path, scope))
    with pytest.raises(ValueError, match="duplicate JSON"): reader.read_parent(PARENT)


def test_codebook_order_not_inherited_or_ignored(tmp_path):
    scope, _ = fixture(tmp_path); path = tmp_path / scope["codebook_paths"][PARENT + "__" + VARIANTS[1]]
    book = json.loads(path.read_text()); book["primitive_ids"].reverse(); write(path, book)
    reader = ReviewNominationReader(tmp_path, scope, reseal(tmp_path, scope))
    with pytest.raises(ValueError, match="codebook/construction"): reader.read_parent(PARENT)
    assert not reader.chunk_reads


def test_exact_rows_and_inventory_hash_both_bound(tmp_path):
    scope, seals = fixture(tmp_path); scope["task_row_summary"][0]["row_count"] = 24
    scope["task_row_summary_sha256"] = digest(scope["task_row_summary"])
    with pytest.raises(ValueError, match="row/identity/count"):
        ReviewNominationReader(tmp_path, scope, seals).read_parent(PARENT)
    scope, seals = fixture(tmp_path); path = tmp_path / next(iter(scope["inventory_source_files"]))
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(ValueError, match="hash drift"):
        ReviewNominationReader(tmp_path, scope, seals).read_parent(PARENT)


def test_scope_leakage_and_missing_codebook_permissions(tmp_path):
    scope, seals = fixture(tmp_path)
    with pytest.raises(PermissionError): ReviewNominationReader(tmp_path, scope, seals).read_parent("S01_synthetic_C08")
    del scope["codebook_paths"]
    with pytest.raises(ValueError, match="explicit distinct codebook"): ReviewNominationReader(tmp_path, scope, seals)


def test_unselected_chunk_not_required_and_source_is_read_only(tmp_path):
    scope, seals = fixture(tmp_path)
    # Unselected third chunk is deliberately absent, even though its old SHA
    # remains in the source seal. Exact selected paths filter it out.
    target(tmp_path, "primitive_index", "2.0").unlink()
    reader = ReviewNominationReader(tmp_path, scope, seals); reader.read_parent(PARENT)
    assert not any(p.endswith("primitive_index/2.0") for p in reader.opened)


def test_duplicate_selected_seal_entry_and_chunk_symlink_fail(tmp_path):
    scope, seals = fixture(tmp_path)
    seal = tmp_path / seals["teacher"]["path"]
    selected = next(line for line in seal.read_text().splitlines() if line.endswith("primitive_index/0.0"))
    seal.write_text(seal.read_text() + selected + "\n"); seals["teacher"]["sha256"] = sha(seal)
    with pytest.raises(ValueError, match="duplicate"):
        ReviewNominationReader(tmp_path, scope, seals).read_parent(PARENT)
    seals = reseal(tmp_path, scope); chunk = target(tmp_path, "primitive_index", "0.0")
    replacement = tmp_path / "external_chunk"; replacement.write_bytes(chunk.read_bytes())
    chunk.unlink(); chunk.symlink_to(replacement)
    with pytest.raises(ValueError, match="symlink"):
        ReviewNominationReader(tmp_path, scope, seals).read_parent(PARENT)


def test_float_group_schema_and_boolean_pose_permission_rejected(tmp_path):
    scope, _ = fixture(tmp_path); path = target(tmp_path, "", ".zgroup")
    write(path, {"zarr_format": 2.0})
    with pytest.raises(ValueError, match="group schema"):
        ReviewNominationReader(tmp_path, scope, reseal(tmp_path, scope)).read_parent(PARENT)
    write(path, {"zarr_format": 2}); attrs = target(tmp_path, "", ".zattrs")
    value = json.loads(attrs.read_text()); value["student_identity_input_forbidden"] = 1; write(attrs, value)
    with pytest.raises(ValueError, match="strict integer/boolean"):
        ReviewNominationReader(tmp_path, scope, reseal(tmp_path, scope)).read_parent(PARENT)
