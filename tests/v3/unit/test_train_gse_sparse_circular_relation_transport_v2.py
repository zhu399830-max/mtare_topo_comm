from pathlib import Path
import inspect

import numpy as np
import torch

import train_gse_axis_anchored_event_relation_v1 as base
import train_gse_sparse_circular_relation_transport_v2 as sparse
from mtare_topo.representation.gse_sparse_circular_relation_transport import (
    MAX_TOKENS,
    SparseCircularRelationTransportNet,
)


ROOT = Path(__file__).resolve().parents[3]
DATASET = ROOT / "results/gate2_representation/gate2_20260824_gse_deduplicated_dataset_export_v1_seed0"


def test_history_rows_are_past_only_and_current_exact() -> None:
    references = np.asarray([[10, 10, 10, 10, 10], [10, 10, 10, 10, 20], [10, 10, 20, 20, 30]])
    rows = sparse._history_rows(references)
    assert rows.tolist() == [[0, 0, 0, 0, 0], [0, 0, 0, 0, 1], [0, 0, 1, 1, 2]]


def test_alignment_is_injective_and_wrap_aware() -> None:
    predicted = torch.tensor([[[179, 1, 40, 90, 120, 150]]])
    identity = torch.full((1, 1, 180), -1)
    identity[0, 0, 0] = 10
    identity[0, 0, 41] = 20
    aligned, teacher_bin = sparse.align_teacher_to_tokens(predicted, identity)
    assert (aligned >= 0).sum() == 2
    assert aligned[0, 0, 0] == 10
    assert aligned[0, 0, 2] == 20
    assert teacher_bin[0, 0, 0] == 0
    assert teacher_bin[0, 0, 2] == 41


def test_real_sparse_core_loss_is_finite() -> None:
    traversals = base.manifest_traversals(DATASET / "artifacts/sequence_manifest.jsonl")
    parent = "S01_flat_tree_small_C01"
    world = sparse._load_world(DATASET / "artifacts/dataset/train", parent, traversals[parent])
    device = torch.device("cpu")
    scans, target = sparse._batch(world, np.asarray([4, 7, 25, 89]), device=device)
    model = SparseCircularRelationTransportNet()
    output = model(scans)
    losses = sparse.sparse_core_loss(output, target)
    losses["total"].backward()
    assert all(torch.isfinite(value) for value in losses.values())
    assert any(parameter.grad is not None for parameter in model.parameters())
    assert target["sequence_presence"].shape == (4, 5, 180)
    assert target["sequence_identity"].shape == (4, 5, 180)
    assert target["sequence_frame_valid"].shape == (4, 5)
    assert target["sequence_pair_valid"].shape == (4, 4)
    assert int(target["sequence_presence"].sum(-1).max()) <= MAX_TOKENS


def test_release_unused_cuda_cache_is_cuda_only(monkeypatch) -> None:
    calls = []
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: calls.append("empty"))
    sparse._release_unused_cuda_cache(torch.device("cpu"))
    assert calls == []
    sparse._release_unused_cuda_cache(torch.device("cuda"))
    assert calls == ["empty"]


def test_cache_release_is_bound_to_training_validation_and_inference_worlds() -> None:
    main_source = inspect.getsource(sparse.main)
    validation_source = inspect.getsource(sparse._validation)
    assert main_source.count("_release_unused_cuda_cache(device)") == 4
    assert validation_source.count("_release_unused_cuda_cache(device)") == 2


def test_nvidia_process_memory_uses_current_pid(monkeypatch) -> None:
    own = sparse.os.getpid()
    monkeypatch.setattr(
        sparse.subprocess,
        "check_output",
        lambda *args, **kwargs: f"999, 7\n{own}, 123\n",
    )
    assert sparse._nvidia_process_memory_bytes() == 123 * 1024**2
