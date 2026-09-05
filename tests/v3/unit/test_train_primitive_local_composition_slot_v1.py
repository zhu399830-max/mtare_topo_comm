import pytest

from train_primitive_local_composition_slot_v1 import (
    EVALUATION_BATCH_SIZE,
    EXPECTED_C07_BATCHES,
    EXPECTED_FIT_BATCHES,
    EXPECTED_HEAD_PARAMETERS,
    EXPECTED_HEAD_TENSORS,
    TRAINING_BATCH_SIZE,
    TRAINING_EPOCHS,
)


def test_frozen_training_schedule_and_parameter_boundary() -> None:
    assert TRAINING_EPOCHS == 3
    assert TRAINING_BATCH_SIZE == EVALUATION_BATCH_SIZE == 128
    assert EXPECTED_FIT_BATCHES == 3411
    assert EXPECTED_C07_BATCHES == 522
    assert EXPECTED_HEAD_PARAMETERS == 1_001_507
    assert EXPECTED_HEAD_TENSORS == 81


def test_schedule_has_exact_expected_optimizer_steps() -> None:
    assert EXPECTED_FIT_BATCHES * TRAINING_EPOCHS == 10_233
