"""Assemble shared candidate inputs and separately held synthetic objectives."""
from dataclasses import dataclass,fields
import numpy as np
import torch
from .gse_candidate_cache_binding import load_bound_candidates
from .gse_synthetic_fit_scope import declared_cases
from mtare_topo.representation.gse_candidate_neighbors import candidate_geometry
from mtare_topo.representation.gse_candidate_objective import candidate_targets
from mtare_topo.evaluation.gse_synthetic_field_scoring import expected_geometry


@dataclass(frozen=True)
class CandidateTrainingExample:
    observation_id: str
    positions_m: torch.Tensor
    context: torch.Tensor
    geometry: object
    target: torch.Tensor
    target_known: torch.Tensor
    expected_positions_m: np.ndarray

    def on_device(self,device):
        g=type(self.geometry)(**{f.name:getattr(self.geometry,f.name).to(device) for f in fields(self.geometry)})
        return (self.positions_m[None].to(device),self.context[None].to(device),g),self.target[None].to(device),self.target_known[None].to(device)


def load_training_examples(root):
    cases={c['case_id']:c for c in declared_cases()}
    examples=[]
    for row in load_bound_candidates(root):
        if not row['context'].supported.all():raise ValueError('unsupported context cannot silently train')
        g=candidate_geometry(row['positions_m'],row['unary'],row['unary_known'])
        truth=expected_geometry(cases[row['observation_id']])
        if not truth['complete_for_declared_fixture']:raise ValueError('fixture completeness mismatch')
        expected=np.asarray(truth['anchors'],dtype=np.float64).reshape(-1,3)
        y,known,missed=candidate_targets(row['positions_m'].numpy(),expected,complete_region=True)
        if missed:raise ValueError('input candidate coverage drift')
        examples.append(CandidateTrainingExample(row['observation_id'],row['positions_m'],row['context'].features,g,
            torch.from_numpy(y),torch.from_numpy(known),expected))
    if len(examples)!=45:raise ValueError('exact45 required')
    return examples
