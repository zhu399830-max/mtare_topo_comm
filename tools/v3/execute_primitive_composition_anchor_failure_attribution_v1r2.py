#!/usr/bin/env python3
"""Execute V1 attribution under the explicit numerical state used by its source evaluator."""

from __future__ import annotations

import torch

import execute_primitive_composition_anchor_failure_attribution_v1 as source


NUMERICAL_CONTRACT = {
    "cuda_matmul_allow_tf32": False,
    "cudnn_allow_tf32": True,
    "float32_matmul_precision": "highest",
    "deterministic_algorithms": False,
}


class _NoOpSetting:
    def __init__(self, wrapped) -> None:
        self._wrapped = wrapped

    def __getattr__(self, name):
        return getattr(self._wrapped, name)

    @property
    def allow_tf32(self):
        return self._wrapped.allow_tf32

    @allow_tf32.setter
    def allow_tf32(self, value) -> None:
        # V1 tries to impose a different contract.  V1R2 retains the explicit
        # source-evaluator state above so exact metric reproduction is meaningful.
        del value


class _BackendsProxy:
    def __init__(self) -> None:
        self.cuda = type("_CudaProxy", (), {})()
        self.cuda.matmul = _NoOpSetting(torch.backends.cuda.matmul)
        self.cudnn = _NoOpSetting(torch.backends.cudnn)


class _TorchProxy:
    backends = _BackendsProxy()

    def __getattr__(self, name):
        return getattr(torch, name)

    @staticmethod
    def use_deterministic_algorithms(mode, *args, **kwargs) -> None:
        del mode, args, kwargs

    @staticmethod
    def set_float32_matmul_precision(precision) -> None:
        del precision


def configure_source_numerics() -> None:
    torch.use_deterministic_algorithms(False)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("highest")
    actual = {
        "cuda_matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
        "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
    }
    if actual != NUMERICAL_CONTRACT:
        raise RuntimeError(f"source-evaluator numerical state drift: {actual}")
    source.torch = _TorchProxy()


def main() -> int:
    configure_source_numerics()
    return source.main()


if __name__ == "__main__":
    raise SystemExit(main())
