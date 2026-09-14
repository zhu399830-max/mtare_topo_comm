# Literature evidence used in the design

- Cano, Tardioli, Mosteo (2026), underground synthetic LiDAR exit CNN and pure topology: https://onlinelibrary.wiley.com/doi/full/10.1002/rob.70157
- Procedural tunnel network generator: https://zaguan.unizar.es/record/148626/files
- TARE official publication page: https://publications.ri.cmu.edu/tare-a-hierarchical-framework-for-efficiently-exploring-complex-3d-environments
- Graph-based subterranean exploration: https://www.research-collection.ethz.ch/entities/publication/076e02d3-f139-4bd1-ac62-9fe87bd0af3c
- PointContrast: https://arxiv.org/abs/2007.10985
- Synthetic supervised plus real unlabeled domain adaptation: https://link.springer.com/article/10.1007/s10514-024-10158-4
- Unsupervised underground junction recognition: https://arxiv.org/abs/2006.04225
- Self-supervised LiDAR traversability: https://arxiv.org/abs/2208.01329
- LAMP 2.0: https://arxiv.org/abs/2205.13135
- SubT-MRS: https://arxiv.org/abs/2307.07607
- Isaac Sim synthetic data generation: https://docs.isaacsim.omniverse.nvidia.com/latest/synthetic_data_generation/index.html
- Isaac Sim RTX LiDAR: https://docs.isaacsim.omniverse.nvidia.com/6.0.1/sensors/isaacsim_sensors_rtx_lidar.html
- Isaac Lab terrain generator: https://isaac-sim.github.io/IsaacLab/develop/source/api/lab/isaaclab.terrains.html

Conclusion: the closest precedent makes “synthetic underground LiDAR -> CNN exits -> topology” insufficient as a novelty claim. This run therefore fixed a planner-consistent objective teacher, causal partial observations, uncertainty, stable exit stubs, M-TARE global replacement, multi-robot allocation, and sealed evaluation as the required differentiators.
