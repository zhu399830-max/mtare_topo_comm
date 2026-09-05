#!/home/zeng-workstation/anaconda3/bin/python
"""Verified-environment replacement entry for AEE sensor export V1.

This wrapper changes only the outer host Python binding.  The collection,
acceptance, failure and sealing implementation remains the frozen V1 runner.
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np
from numpy._core import _multiarray_umath

import run_aee_domain_sensor_export_v1 as implementation


RUN_ID = "gate2_20260820_aee_domain_sensor_export_v1r_seed20260820"
EXPECTED_PYTHON = Path("/home/zeng-workstation/anaconda3/bin/python").resolve()
EXPECTED_PYTHON_SHA256 = "c3d907b2ac9c12d0ba5777f8015dd25df1b63979002144b35bfdde0fbd32efb6"
EXPECTED_NUMPY_VERSION = "2.1.3"
EXPECTED_NUMPY_INIT_SHA256 = "39c42db027548f958e096e8babe3fa0e3e773d24aa39eb6363fc0e3abbec34b1"
EXPECTED_NUMPY_CORE_SHA256 = "f21b205fff55303468014bdb9aa50345930b5782b441ac8753d73e0e2aa8a134"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_host_python() -> None:
    actual = Path(sys.executable).resolve()
    if actual != EXPECTED_PYTHON:
        raise RuntimeError(
            f"replacement requires {EXPECTED_PYTHON}, observed {actual}"
        )
    if sha256(actual) != EXPECTED_PYTHON_SHA256:
        raise RuntimeError("host Python binary identity drift")
    if np.__version__ != EXPECTED_NUMPY_VERSION:
        raise RuntimeError(
            f"NumPy identity drift: expected {EXPECTED_NUMPY_VERSION}, observed {np.__version__}"
        )
    if sha256(Path(np.__file__).resolve()) != EXPECTED_NUMPY_INIT_SHA256:
        raise RuntimeError("NumPy package identity drift")
    if sha256(Path(_multiarray_umath.__file__).resolve()) != EXPECTED_NUMPY_CORE_SHA256:
        raise RuntimeError("NumPy core binary identity drift")


def main() -> int:
    verify_host_python()
    implementation.RUN_ID = RUN_ID
    return implementation.main()


if __name__ == "__main__":
    raise SystemExit(main())
