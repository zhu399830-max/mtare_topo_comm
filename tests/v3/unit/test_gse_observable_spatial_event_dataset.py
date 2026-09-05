from __future__ import annotations

import numpy as np
import pytest

from mtare_topo.data.gse_observable_spatial_event_dataset import (
    ObservableSpatialEventTeacherArrays,
)


def test_observable_teacher_array_contract_fails_closed_on_wrong_population() -> None:
    with pytest.raises(RuntimeError, match="contract drift"):
        ObservableSpatialEventTeacherArrays(
            global_sequence_index=np.arange(2, dtype=np.int64),
            partition_code=np.zeros(2, dtype=np.uint8),
            event_type_index=np.full((2, 16), -1, dtype=np.int8),
            event_relative_xyz_m=np.zeros((2, 16, 3), dtype=np.float32),
            event_identity_index=np.full((2, 16), -1, dtype=np.int32),
            event_mask=np.zeros((2, 16), dtype=np.uint8),
            set_cardinality=np.zeros(2, dtype=np.uint8),
            parent_id=np.asarray(("a", "b")),
        )
