# Isaac Sim 6.0.1 Compatibility Checker

Executed at 2026-08-10T07:55:28Z with the pinned image and repository digest recorded in `artifacts/isaac_image_audit.json`.

Command policy: temporary container, GPU enabled, `ACCEPT_EULA=Y`, `--network=none`, automatic removal, no privacy-consent opt-in.

Result: `System checking result: PASSED`; process exit code 0.

Detected:

- NVIDIA GeForce RTX 5090 D; driver 580.173.02; supported.
- VRAM 34.19 GB; good.
- Intel Core i9-14900K; 32 available cores; excellent.
- RAM 67.17 GB; good.
- Storage 1332.75 GB reported available in the container; excellent.
- Ubuntu 24.04.3 LTS; supported.

Non-blocking warnings:

- CPU power governor is `powersave`; performance is recommended before timed batch generation.
- No display was detected; expected for the selected headless route.
- OmniHub could not start/reconnect under `--network=none`; the local hardware checker still completed and passed.

This pass authorizes only the next infrastructure implementation stage. It is not a LiDAR, USD import, sensor parity, dataset, or training result.
