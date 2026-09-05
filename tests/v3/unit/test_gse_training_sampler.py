from __future__ import annotations

import unittest

from mtare_topo.data.gse_training_sampler import PairAwareBatchSampler


class PairAwareBatchSamplerTest(unittest.TestCase):
    def test_every_sample_once_and_positive_units_stay_together(self) -> None:
        labels = [-1, 10, 10, 10, 20, 20, -1, 30]
        sampler = PairAwareBatchSampler(labels, batch_size=4, seed=7)
        batches = list(sampler)
        flat = [index for batch in batches for index in batch]
        self.assertEqual(sorted(flat), list(range(len(labels))))
        self.assertEqual(len(flat), len(set(flat)))
        positions = {index: batch_index for batch_index, batch in enumerate(batches) for index in batch}
        paired_ten = sum(positions[first] == positions[second] for first in (1, 2, 3) for second in (1, 2, 3) if first < second)
        self.assertGreaterEqual(paired_ten, 1)
        self.assertEqual(positions[4], positions[5])

    def test_epoch_is_deterministic_but_changes_order(self) -> None:
        sampler = PairAwareBatchSampler([-1, 1, 1, 2, 2, -1, 3, 3], batch_size=4, seed=2)
        first = list(sampler)
        self.assertEqual(first, list(sampler))
        sampler.set_epoch(1)
        self.assertNotEqual(first, list(sampler))

    def test_invalid_contract_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            PairAwareBatchSampler([], batch_size=4, seed=0)
        with self.assertRaises(ValueError):
            PairAwareBatchSampler([1, 1], batch_size=3, seed=0)


if __name__ == "__main__":
    unittest.main()
