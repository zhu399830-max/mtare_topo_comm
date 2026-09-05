from __future__ import annotations

import numpy as np
import torch

from mtare_topo.representation.gse_typed_composers import ActionSetRelationComposer, MetricChangeComposer
from train_gse_dual_composer_v1 import Bundle, _geometry_stats, action_input, infer, metric_input


def _bundle(rows: int = 8) -> Bundle:
    rng=np.random.default_rng(4);shape=(rows,5,6)
    count=rng.random((rows,5,7),dtype=np.float32);count/=count.sum(-1,keepdims=True)
    transport=rng.random((rows,4,6,7),dtype=np.float32);transport/=transport.sum(-1,keepdims=True)
    event=np.asarray(["corridor","junction","terminal","turn","geometry_transition","corridor","junction","turn"])
    return Bundle(
        global_index=np.arange(rows,dtype=np.int64),parent=np.full(rows,"S01_flat_tree_small_C01"),event=event,identity=np.asarray(["","j0","t0","u0","g0","","j1","u1"]),
        action_target=np.asarray([0,1,2,0,0,0,1,0],dtype=np.int8),metric_target=np.asarray([0,0,0,1,2,0,0,1],dtype=np.int8),mask=np.ones((rows,5),dtype=bool),back_steps=np.asarray([0,0,0,1.5,2.5,0,0,1],dtype=np.float32),back_valid=np.asarray([0,0,0,1,1,0,0,1],dtype=bool),
        bearing=rng.uniform(-180,180,shape).astype(np.float32),existence=rng.normal(size=shape).astype(np.float32),opening=rng.uniform(.5,5,shape).astype(np.float32),profile=rng.uniform(.2,4,shape+(4,)).astype(np.float32),token_uncertainty=rng.uniform(.01,1,shape+(5,)).astype(np.float32),count=count,transport=transport,reveal=rng.random((rows,4,6),dtype=np.float32),geometry=np.concatenate((rng.uniform(1,6,(rows,5,2)),rng.uniform(-8,8,(rows,5,1)),rng.uniform(0,.1,(rows,5,1))),axis=-1).astype(np.float32),geometry_uncertainty=rng.uniform(.01,1,(rows,5,4)).astype(np.float32),
    )


def test_trainer_batch_builders_and_all_ablations_are_finite() -> None:
    bundle=_bundle();device=torch.device("cpu");index=np.arange(len(bundle))
    action=action_input(bundle,index,device);metric=metric_input(bundle,index,device)
    assert action.token_bearing_unit.shape==(8,5,6,2) and metric.geometry_sequence.shape==(8,5,4)
    output=infer(ActionSetRelationComposer(),MetricChangeComposer(),bundle,device,4,_geometry_stats(bundle))
    assert set(output)=={"full","no_metric","no_transport","single_frame","count_bearing_only"}
    for values in output.values():
        assert values["logits"].shape==(8,5)
        assert np.isfinite(values["logits"]).all()
        np.testing.assert_allclose(np.exp(values["logits"]).sum(-1),1.0,atol=1e-5)
