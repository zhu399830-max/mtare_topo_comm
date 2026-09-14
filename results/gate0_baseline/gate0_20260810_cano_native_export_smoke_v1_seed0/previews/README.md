# Preview provenance

- Run: `gate0_20260810_cano_native_export_smoke_v1_seed0`
- Split: `NONE_NATIVE_EXPORT_SMOKE_ONLY`; formal dataset world count is zero.
- World: original Cano temporary export `env_001`; no trajectory or LiDAR sample exists.
- File: `native_mesh_axis_diagnostic.png`.
- Method: deterministic X-Y, X-Z, and Y-Z projections of all 61,421 exported OBJ vertices and all 4,382 exported axis rows.
- Coordinates: source XYZ units, expected metres from the upstream generator; no independent scale certification was performed.
- Grey: mesh vertices; blue: `axis.txt` tunnel rows; red: `axis.txt` intersection rows.
- Supports: visual audit of complete exported extent and approximate mesh-axis alignment.
- Does not support: topology-label validity, graph-mesh consistency, watertightness, navigation feasibility, Isaac rendering, dataset quality, or learning readiness.
- Machine-readable evidence: `metrics/native_export_validation.json`.
