#!/usr/bin/env python3
"""Zero-dataset surface-to-axis software contract; no trained checkpoint."""
from run_gse_composition_software_contract_v1 import main

if __name__ == "__main__":
    raise SystemExit(main(limitations=[
        "Synthetic tensor tests only; no learned accuracy, unique point ownership or scientific Gate PASS.",
        "One full-size random-initialized legacy-backbone adapter test, zero checkpoint/data/optimizer.",
        "Vote operator is SE(3)-equivariant with vector offsets; neural coordinate MLP/backbone is not claimed equivariant.",
        "Direction-defined, membership mass, ESS and vote spread are diagnostics, not calibrated uncertainty or port evidence.",
        "Multi-source returns need distributional ownership/slot-conditioned supervision; no teacher was generated.",
        "Actual frozen-weight reuse, geometry learning and event/port supervision remain unverified for this new branch.",
    ]))
