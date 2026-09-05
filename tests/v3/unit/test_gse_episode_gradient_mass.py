from __future__ import annotations

import numpy as np

from mtare_topo.evaluation.gse_episode_gradient_mass import analytic_episode_class_mass


def test_analytic_episode_mass_exposes_rare_class_multiplier() -> None:
    probability = np.asarray([
        [.1, .7, .1, .05, .05], [.1, .8, .04, .03, .03],
        [.1, .7, .1, .05, .05], [.1, .8, .04, .03, .03],
        [.1, .7, .1, .05, .05], [.1, .05, .05, .75, .05],
        [.1, .05, .05, .05, .75], [.1, .05, .75, .05, .05],
    ])
    target = np.asarray([1, 1, 1, 1, 1, 3, 4, 2])
    episode = np.asarray([0, 0, 1, 1, 2, 3, 4, 5])
    result = analytic_episode_class_mass(probability, target, episode)
    assert result["episodes"] == 6
    junction = result["per_event"]["junction"]
    transition = result["per_event"]["geometry_transition"]
    assert junction["episodes"] == 3
    assert transition["episodes"] == 1
    assert junction["raw_episode_mass_share"] == .5
    assert transition["equal_class_per_episode_weight_multiplier"] == 1.5
