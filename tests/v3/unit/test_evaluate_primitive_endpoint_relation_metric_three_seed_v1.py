import numpy as np
import torch

from evaluate_primitive_endpoint_relation_metric_three_seed_v1 import RelationBuffer, complete_link_metrics


def test_relation_buffer_preserves_rows_endpoints_and_missing_positive_denominator() -> None:
    score = torch.zeros(1, 4, 4); target = torch.zeros(1, 4, 4, dtype=torch.bool); overlap = target.clone(); eligible = target.clone()
    score[0, 0, 2] = .9; target[0, 0, 2] = True; eligible[0, 0, 2] = True
    score[0, 1, 3] = .2; overlap[0, 1, 3] = True; eligible[0, 1, 3] = True
    buffer = RelationBuffer(); buffer.append(score, target, overlap, eligible, 7)
    metrics, arrays = buffer.finalize(total_positive=2)
    assert arrays["row"].tolist() == [7, 7]
    assert arrays["left"].tolist() == [0, 1] and arrays["right"].tolist() == [2, 3]
    assert metrics["best_f1"]["false_negative"] >= 1


def test_complete_link_removes_chain_merge() -> None:
    arrays = {
        "score": np.asarray([.9, .2, .9, .8], np.float32),
        "target": np.asarray([True, False, False, True]),
        "overlap": np.asarray([False, True, False, False]),
        "row": np.asarray([0, 0, 0, 0], np.int32),
        "left": np.asarray([0, 0, 1, 3], np.uint8),
        "right": np.asarray([1, 2, 2, 4], np.uint8),
    }
    result = complete_link_metrics(arrays, .7, total_positive=2)
    assert result["predicted_clusters"] == 2
    assert result["true_positive"] == 2
    assert result["false_positive"] == 0
    assert result["overlap_false_positive"] == 0
