import hashlib
import io
import numpy as np
import pytest
import torch
from mtare_topo.representation.block_relation_training_v1 import build_relation_model, state_sha256, forward_observation
from mtare_topo.representation.frozen_relation_inference_v1 import load_final_model, predict_r2, StudentObservation
from mtare_topo.representation.gse_block_points import bind_block_points


def checkpoint(**changes):
    model = build_relation_model(relation_attributes=True, device='cpu')
    data = dict(model=model.state_dict(), optimizer={}, updates=2000, initial_sha256='a'*64)
    data.update(changes)
    stream = io.BytesIO(); torch.save(data, stream)
    return stream.getvalue()


def load(payload, **kwargs):
    return load_final_model(payload, expected_sha256=hashlib.sha256(payload).hexdigest(),
                            initial_sha256='a'*64, relation_attributes=True, **kwargs)


def test_checkpoint_rejection_and_exact_frozen_state():
    torch.set_num_threads(1)
    payload = checkpoint()
    model = load(payload)
    assert not model.training and not any(p.requires_grad for p in model.parameters())
    assert state_sha256(model) == state_sha256(build_relation_model(relation_attributes=True))
    with pytest.raises(ValueError, match='hash'):
        load_final_model(payload, expected_sha256='0'*64, initial_sha256='a'*64, relation_attributes=True)
    with pytest.raises(ValueError, match='2000'):
        load(checkpoint(updates=1999))
    with pytest.raises(ValueError, match='2000'):
        load(checkpoint(initial_sha256='b'*64))


def test_inference_reuses_original_forward_without_teacher_or_mutation():
    torch.set_num_threads(1)
    model = load(checkpoint())
    xyz = np.array([[0,0,0],[0,1,0],[2,0,0],[2,1,0]], np.float32)
    blocks = bind_block_points(xyz, np.array([0,1,0,1]), np.array([0,0,1,1]))
    student = dict(blocks=blocks, context=np.zeros((2,128), np.float32), binding=None)
    before = state_sha256(model)
    a = predict_r2(model, student); b = predict_r2(model, student)
    with torch.inference_mode():
        original = forward_observation(model, StudentObservation({'r2': student}), 'r2')
    for key in a:
        np.testing.assert_array_equal(a[key], b[key])
        np.testing.assert_array_equal(a[key], getattr(original,key).numpy())
    assert state_sha256(model) == before
    assert all(p.grad is None for p in model.parameters())
    with pytest.raises(ValueError, match='only observation'):
        predict_r2(model, dict(student, teacher={}))
    model.train()
    with pytest.raises(ValueError, match='frozen'):
        predict_r2(model, student)
