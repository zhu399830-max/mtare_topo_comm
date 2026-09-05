"""Validation and enumeration for the paired Gate-5 development matrix."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import random
from typing import Any


FAMILIES = ("original_mtare", "m1d_topology", "layered_gt_map_oracle")


@dataclass(frozen=True)
class ClosedLoopCase:
    index: int
    world: str
    environment_seed: int
    method_id: str
    method_family: str
    checkpoint_seed: int | None
    runtime_sec: int

    @property
    def case_id(self) -> str:
        return f"{self.index:03d}_{self.world}_env{self.environment_seed}_{self.method_id}"

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "case_id": self.case_id}


def load_development_matrix(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("schema_version") != "mtare_single_robot_development_matrix_v1":
        raise ValueError("unexpected development-matrix schema")
    worlds = payload.get("worlds", [])
    seeds = payload.get("environment_seeds", [])
    methods = payload.get("method_variants", [])
    runtime = payload.get("runtime_sec")
    if [world.get("name") for world in worlds] != ["tunnel", "garage"]:
        raise ValueError("development worlds or ordering drift")
    if len(seeds) != 5 or len(set(seeds)) != 5 or any(not isinstance(seed, int) or seed < 0 for seed in seeds):
        raise ValueError("exactly five distinct non-negative environment seeds are required")
    if [method.get("id") for method in methods] != [
        "original_mtare",
        "m1d_seed0",
        "m1d_seed1",
        "m1d_seed2",
        "layered_gt_map_oracle",
    ]:
        raise ValueError("method identity or ordering drift")
    if tuple(dict.fromkeys(method["family"] for method in methods)) != FAMILIES:
        raise ValueError("method family drift")
    for expected_seed, method in enumerate(methods[1:4]):
        if method.get("checkpoint_seed") != expected_seed or not method.get("checkpoint_path") or len(method.get("checkpoint_sha256", "")) != 64:
            raise ValueError("M1D checkpoint identity is incomplete")
    if not isinstance(runtime, int) or runtime <= 0:
        raise ValueError("runtime must be a positive integer number of seconds")
    for world in worlds:
        start = world.get("start", {})
        values = tuple(start.get(key) for key in ("vehicle_x_m", "vehicle_y_m", "terrain_z_m", "vehicle_yaw_deg"))
        if any(not isinstance(value, (int, float)) or not math.isfinite(value) for value in values):
            raise ValueError("world start pose must be finite")
        if len(world.get("world_sha256", "")) != 64 or len(world.get("complete_map_sha256", "")) != 64:
            raise ValueError("world/map hashes are not frozen")
    expected = len(worlds) * len(seeds) * len(methods)
    pairing = payload.get("pairing", {})
    if pairing.get("case_count") != expected or pairing.get("simulated_runtime_sec") != expected * runtime:
        raise ValueError("pairing count or simulated runtime drift")
    if payload.get("blocker", {}).get("status") != "UNRESOLVED_REQUIRES_USER_APPROVAL":
        raise ValueError("proposal must not conceal the original-TARE seed blocker")
    return payload


def enumerate_cases(matrix: dict[str, Any]) -> tuple[ClosedLoopCase, ...]:
    cases: list[ClosedLoopCase] = []
    for world in matrix["worlds"]:
        for environment_seed in matrix["environment_seeds"]:
            for method in matrix["method_variants"]:
                cases.append(
                    ClosedLoopCase(
                        index=len(cases),
                        world=world["name"],
                        environment_seed=environment_seed,
                        method_id=method["id"],
                        method_family=method["family"],
                        checkpoint_seed=method["checkpoint_seed"],
                        runtime_sec=matrix["runtime_sec"],
                    )
                )
    if len({case.case_id for case in cases}) != len(cases):
        raise RuntimeError("closed-loop case IDs are not unique")
    return tuple(cases)


def paired_case_audit(cases: tuple[ClosedLoopCase, ...]) -> dict[str, Any]:
    pairs: dict[tuple[str, int], list[ClosedLoopCase]] = {}
    for case in cases:
        pairs.setdefault((case.world, case.environment_seed), []).append(case)
    expected_methods = ("original_mtare", "m1d_seed0", "m1d_seed1", "m1d_seed2", "layered_gt_map_oracle")
    complete = all(tuple(case.method_id for case in group) == expected_methods for group in pairs.values())
    return {
        "schema_version": "mtare_single_robot_pairing_audit_v1",
        "case_count": len(cases),
        "paired_world_seed_count": len(pairs),
        "methods_per_pair": len(expected_methods),
        "all_pairs_complete_and_ordered": complete,
        "total_simulated_runtime_sec": sum(case.runtime_sec for case in cases),
    }


__all__ = ["ClosedLoopCase", "enumerate_cases", "load_development_matrix", "paired_case_audit"]


@dataclass(frozen=True)
class StochasticClosedLoopCase:
    index: int
    world: str
    environment_seed: int
    method_id: str
    method_family: str
    checkpoint_seed: int | None
    execution_repeat: int
    runtime_sec: int

    @property
    def case_id(self) -> str:
        return (
            f"{self.index:03d}_{self.world}_env{self.environment_seed}_"
            f"{self.method_id}_repeat{self.execution_repeat}"
        )

    @property
    def block_id(self) -> str:
        return f"{self.world}_env{self.environment_seed}"

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "case_id": self.case_id, "block_id": self.block_id}


def load_stochastic_matrix(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("schema_version") != "mtare_single_robot_stochastic_matrix_v1":
        raise ValueError("unexpected stochastic-matrix schema")
    if payload.get("status") != "APPROVED_STOCHASTIC_DESIGN":
        raise ValueError("stochastic design is not approved")
    worlds = payload.get("worlds", [])
    seeds = payload.get("environment_seeds", [])
    if [world.get("name") for world in worlds] != ["tunnel", "garage"]:
        raise ValueError("stochastic development worlds or ordering drift")
    if seeds != [11, 23, 37, 53, 71]:
        raise ValueError("stochastic environment seeds drift")
    if payload.get("runtime_sec") != 600 or payload.get("schedule_seed") != 20260820:
        raise ValueError("runtime or randomization schedule seed drift")
    design = payload.get("family_design", {})
    if design.get("original_mtare", {}).get("execution_repeats") != [0, 1, 2]:
        raise ValueError("original M-TARE repeat contract drift")
    if design.get("m1d_topology", {}).get("checkpoint_seeds") != [0, 1, 2]:
        raise ValueError("M1D checkpoint replicate contract drift")
    if design.get("m1d_topology", {}).get("execution_repeats") != [0]:
        raise ValueError("M1D execution-repeat contract drift")
    if design.get("layered_gt_map_oracle", {}).get("execution_repeats") != [0, 1, 2]:
        raise ValueError("Oracle repeat contract drift")
    checkpoints = payload.get("m1d_checkpoints", [])
    if [item.get("seed") for item in checkpoints] != [0, 1, 2]:
        raise ValueError("M1D checkpoint seed order drift")
    if any(not item.get("path") or len(item.get("sha256", "")) != 64 for item in checkpoints):
        raise ValueError("M1D checkpoint identity incomplete")
    for world in worlds:
        if len(world.get("world_sha256", "")) != 64 or len(world.get("complete_map_sha256", "")) != 64:
            raise ValueError("world/map hash identity incomplete")
        start = world.get("start", {})
        values = tuple(start.get(key) for key in ("vehicle_x_m", "vehicle_y_m", "terrain_z_m", "vehicle_yaw_deg"))
        if any(not isinstance(value, (int, float)) or not math.isfinite(value) for value in values):
            raise ValueError("world start pose must be finite")
    expected = len(worlds) * len(seeds) * 9
    if payload.get("case_count") != expected or payload.get("simulated_runtime_sec") != expected * 600:
        raise ValueError("stochastic matrix count/runtime drift")
    return payload


def enumerate_stochastic_cases(matrix: dict[str, Any]) -> tuple[StochasticClosedLoopCase, ...]:
    unordered: list[dict[str, Any]] = []
    checkpoint_by_seed = {int(item["seed"]): item for item in matrix["m1d_checkpoints"]}
    for world in matrix["worlds"]:
        for environment_seed in matrix["environment_seeds"]:
            for repeat in matrix["family_design"]["original_mtare"]["execution_repeats"]:
                unordered.append(
                    dict(world=world["name"], environment_seed=environment_seed, method_id="original_mtare",
                         method_family="original_mtare", checkpoint_seed=None, execution_repeat=repeat)
                )
            for seed in matrix["family_design"]["m1d_topology"]["checkpoint_seeds"]:
                if seed not in checkpoint_by_seed:
                    raise ValueError("matrix references an absent M1D checkpoint")
                unordered.append(
                    dict(world=world["name"], environment_seed=environment_seed, method_id=f"m1d_seed{seed}",
                         method_family="m1d_topology", checkpoint_seed=seed, execution_repeat=0)
                )
            for repeat in matrix["family_design"]["layered_gt_map_oracle"]["execution_repeats"]:
                unordered.append(
                    dict(world=world["name"], environment_seed=environment_seed, method_id="layered_gt_map_oracle",
                         method_family="layered_gt_map_oracle", checkpoint_seed=None, execution_repeat=repeat)
                )
    random.Random(int(matrix["schedule_seed"])).shuffle(unordered)
    cases = tuple(
        StochasticClosedLoopCase(index=index, runtime_sec=int(matrix["runtime_sec"]), **item)
        for index, item in enumerate(unordered)
    )
    if len(cases) != 90 or len({case.case_id for case in cases}) != 90:
        raise RuntimeError("stochastic case schedule must contain 90 unique cases")
    return cases


def stochastic_case_audit(cases: tuple[StochasticClosedLoopCase, ...]) -> dict[str, Any]:
    blocks: dict[str, list[StochasticClosedLoopCase]] = {}
    for case in cases:
        blocks.setdefault(case.block_id, []).append(case)
    complete = True
    for group in blocks.values():
        by_family: dict[str, list[StochasticClosedLoopCase]] = {}
        for case in group:
            by_family.setdefault(case.method_family, []).append(case)
        complete &= set(by_family) == set(FAMILIES)
        complete &= sorted(item.execution_repeat for item in by_family.get("original_mtare", [])) == [0, 1, 2]
        complete &= sorted(item.checkpoint_seed for item in by_family.get("m1d_topology", [])) == [0, 1, 2]
        complete &= sorted(item.execution_repeat for item in by_family.get("layered_gt_map_oracle", [])) == [0, 1, 2]
    family_counts = {family: sum(case.method_family == family for case in cases) for family in FAMILIES}
    return {
        "schema_version": "mtare_single_robot_stochastic_matrix_audit_v1",
        "case_count": len(cases),
        "block_count": len(blocks),
        "cases_per_block": 9,
        "family_case_counts": family_counts,
        "all_blocks_complete": bool(complete),
        "total_simulated_runtime_sec": sum(case.runtime_sec for case in cases),
        "schedule_sha256": __import__("hashlib").sha256(
            json.dumps([case.to_dict() for case in cases], sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
    }


__all__ += [
    "StochasticClosedLoopCase",
    "enumerate_stochastic_cases",
    "load_stochastic_matrix",
    "stochastic_case_audit",
]
