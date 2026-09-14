# Source audit

Primary technical sources used for this design:

- Lorenzo Cano et al., “Procedural Generation of Underground Environments for Gazebo” (2022), which identifies the legacy Gazebo/SubT-tile repository: <https://github.com/LorenzoCanoAn/gzb-subt-env-proc-gen>
- Lorenzo Cano et al., “Procedural generation of tunnel networks for data collection and subterranean navigation” (IROS 2024), used for the graph-to-mesh generation concept. A separate public code repository was not located in this audit.
- Lorenzo Cano et al., Journal of Field Robotics (2026), used as the closest perception/navigation baseline and source of the reported 50-world/500,000-sample precedent.
- NVIDIA Isaac Sim workstation installation documentation: <https://docs.isaacsim.omniverse.nvidia.com/latest/installation/install_workstation.html>
- NVIDIA Isaac Sim container installation documentation: <https://docs.isaacsim.omniverse.nvidia.com/latest/installation/install_container.html>
- IEEE RA-L journal page: <https://www.ieee-ras.org/publications/ieee-robotics-and-automation-letters/>
- Journal of Field Robotics journal page and author guide: <https://onlinelibrary.wiley.com/journal/15564967> and <https://onlinelibrary.wiley.com/page/journal/15564967/homepage/guide.htm>
- IEEE Transactions on Robotics journal page and author information: <https://www.ieee-ras.org/publications/t-ro/> and <https://www.ieee-ras.org/publications/t-ro/t-ro-information-for-authors/>

Unresolved provenance:

- The legacy GitHub repository's current commit, LICENSE file, dependency state, and compatibility were not retrieved because network repository inspection did not complete.
- No public repository for the 2024 graph-to-mesh implementation was identified.
- Therefore no third-party generator code is approved for integration yet.
