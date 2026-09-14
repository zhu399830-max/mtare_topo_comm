"""Paired P1a/P1b loader with the compact endpoint-observability sidecar."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import zarr

from mtare_topo.data.primitive_attachment_observability_sidecar import (
    PRIMARY_SUPPORT_BAND_M,
    unpack_endpoint_observed,
)
from mtare_topo.data.primitive_relation_batches import (
    PrimitiveRelationBatchLoader,
    PrimitiveRelationNumpyBatch,
)


@dataclass(frozen=True)
class ObservablePrimitiveRelationNumpyBatch:
    base: PrimitiveRelationNumpyBatch
    endpoint_observed: np.ndarray

    def __post_init__(self) -> None:
        batch = len(self.base.range_valid)
        observed = np.asarray(self.endpoint_observed)
        if observed.shape != (batch, 32, 2) or observed.dtype != np.uint8:
            raise ValueError("endpoint observed batch must be uint8 [B,32,2]")
        if np.any((observed != 0) & (observed != 1)):
            raise ValueError("endpoint observed batch must be binary")
        if np.any(observed.astype(bool) & ~self.base.primitive_mask.astype(bool)[:, :, None]):
            raise ValueError("inactive primitive endpoint cannot be observed")


class ObservablePrimitiveRelationBatchLoader(PrimitiveRelationBatchLoader):
    """Read immutable sensor/Teacher shards plus aligned support bits."""

    def __init__(self, sensor_root: Path, teacher_root: Path, sidecar_root: Path) -> None:
        super().__init__(sensor_root, teacher_root)
        self.sidecar_root = Path(sidecar_root)
        sidecar_tasks = {value.name for value in self.sidecar_root.glob("*.zarr") if value.is_dir()}
        if sidecar_tasks != set(self.task_names):
            raise ValueError("endpoint observability sidecars are empty or task-misaligned")
        for name in self.task_names:
            sidecar = zarr.open_group(str(self.sidecar_root / name), mode="r")
            if sidecar.attrs.get("schema_version") != "primitive_attachment_observability_sidecar_v1":
                raise ValueError("endpoint observability sidecar schema drift")
            if float(sidecar.attrs.get("support_band_m", -1.0)) != PRIMARY_SUPPORT_BAND_M:
                raise ValueError("endpoint observability support band drift")
            if sidecar.attrs.get("hidden_pair_semantics") != "unknown_never_negative":
                raise ValueError("endpoint observability unknown-pair contract drift")
            if int(sidecar.attrs.get("endpoint_count", -1)) != 64 or int(sidecar.attrs.get("packed_endpoint_bytes", -1)) != 8:
                raise ValueError("endpoint observability bit capacity drift")
            length = self._lengths[name]
            if sidecar["endpoint_observed_packed"].shape != (length, 8) or sidecar["source_global_sequence_index"].shape != (length,):
                raise ValueError("endpoint observability sequence shape drift")
            teacher = zarr.open_group(str(self.teacher_root / name), mode="r")
            if not np.array_equal(sidecar["source_global_sequence_index"][:], teacher["source_global_sequence_index"][:]):
                raise ValueError("endpoint observability source sequence identity drift")

    def _read(self, name: str, indices: np.ndarray) -> ObservablePrimitiveRelationNumpyBatch:
        base = super()._read(name, indices)
        sidecar = zarr.open_group(str(self.sidecar_root / name), mode="r")
        packed = np.asarray(sidecar["endpoint_observed_packed"].oindex[np.asarray(indices, dtype=np.int64)], dtype=np.uint8)
        observed = unpack_endpoint_observed(packed)
        return ObservablePrimitiveRelationNumpyBatch(base=base, endpoint_observed=observed)


__all__ = ["ObservablePrimitiveRelationBatchLoader", "ObservablePrimitiveRelationNumpyBatch"]
