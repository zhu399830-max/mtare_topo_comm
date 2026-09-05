# 单 TNG→Mesh→Isaac 导入：运行方法

状态：已按 2026-08-10 固定环境验证。以下命令都从项目根目录 `/home/zeng-workstation/mtare_topo_comm` 执行。

## 1. 直接查看本次已完成结果

- 完整实际网格投影：`results/gate0_baseline/gate0_20260810_isaac_generator_compatibility_smoke_v1_seed0/previews/tunnel_mesh_g000_actual_surface_projections.png`
- 网格验收：`results/gate0_baseline/gate0_20260810_isaac_generator_compatibility_smoke_v1_seed0/artifacts/tng_3eca286d5e4c4cd3_g000_mesh_v1/mesh_validation.json`
- Isaac 导入验收：`results/gate0_baseline/gate0_20260810_isaac_generator_compatibility_smoke_v1_seed0/logs/isaac_import_v2_pass.json`
- 可导入 USD：`results/gate0_baseline/gate0_20260810_isaac_generator_compatibility_smoke_v1_seed0/artifacts/tng_3eca286d5e4c4cd3_g000_mesh_v1/isaac_stage_v2.usda`

## 2. 先跑测试

```bash
cd /home/zeng-workstation/mtare_topo_comm
python3 -m unittest discover -s tests/v3/unit -p 'test_*.py'
```

预期：40 个测试全部通过。

## 3. 重放生成一张新副本

生成器采用不可覆盖输出，下面先选一个新的手工运行目录；如果该目录已经存在，请换一个新名字。

```bash
cd /home/zeng-workstation/mtare_topo_comm
mkdir -p results/manual_runs/tng_mesh_replay_001
python3 tools/v3/worldgen/generate_tng.py \
  --config configs/v3/gate0/worldgen/tng_mesh_aware_smoke_v1.json \
  --output results/manual_runs/tng_mesh_replay_001/topology.json

python3 tools/v3/worldgen/generate_tunnel_mesh.py \
  --topology results/manual_runs/tng_mesh_replay_001/topology.json \
  --config configs/v3/gate0/worldgen/tunnel_mesh_smoke_g000_v1.json \
  --output-dir results/manual_runs/tng_mesh_replay_001/mesh

python3 tools/v3/worldgen/render_mesh_preview.py \
  --obj results/manual_runs/tng_mesh_replay_001/mesh/collision_render_mesh.obj \
  --topology results/manual_runs/tng_mesh_replay_001/topology.json \
  --output results/manual_runs/tng_mesh_replay_001/actual_mesh_projection.png
```

预期 topology parent 为 `tng_3eca286d5e4c4cd3`，mesh hash 为 `274fe9e504ef5de8b9cea842c26bcaf343196cbd8d5c217973850a55cba547ed`。新导出的 USD 文件名通常是 `isaac_stage.usda`；它使用当前已修复的 serializer，与历史失败文件不是同一内容。

## 4. 在官方 Isaac Sim 6.0.1 容器中验证 USD

先创建容器可写的结果目录：

```bash
mkdir -p /tmp/mtare_isaac_validation
chmod 777 /tmp/mtare_isaac_validation
```

验证本次冻结的 v2 USD：

```bash
cd /home/zeng-workstation/mtare_topo_comm
docker run --entrypoint bash --gpus all --rm --network=none \
  -e ACCEPT_EULA=Y \
  -v "$PWD":/workspace:ro \
  -v /tmp/mtare_isaac_validation:/output:rw \
  nvcr.io/nvidia/isaac-sim:6.0.1 \
  ./python.sh /workspace/tools/v3/isaac/validate_usd_import.py \
  --usd /workspace/results/gate0_baseline/gate0_20260810_isaac_generator_compatibility_smoke_v1_seed0/artifacts/tng_3eca286d5e4c4cd3_g000_mesh_v1/isaac_stage_v2.usda \
  --result /output/isaac_import_validation.json \
  --updates 10
```

成功标志：终端打印的机器结果中 `status` 为 `PASS`，进程退出码为 0。结果保存在 `/tmp/mtare_isaac_validation/isaac_import_validation.json`。

## 5. 当前边界

这个命令只验证 USD 可打开、mesh 数量/extent/单位和 collision API；它不证明机器人可通行，也不采 LiDAR。本轮 Isaac headless Replicator 截图失败，因此查看地图应使用实际 OBJ 投影图；不要把它称为 Isaac RTX 截图。

## 6. 当前 g001 navigation-grade 结果

生成当前导航母图和几何时使用：

```bash
python3 tools/v3/worldgen/generate_tng.py \
  --config configs/v3/gate0/worldgen/tng_mesh_aware_smoke_v1.json \
  --output results/manual_runs/navigation_replay_001/topology.json

python3 tools/v3/worldgen/generate_tunnel_mesh.py \
  --topology results/manual_runs/navigation_replay_001/topology.json \
  --config configs/v3/gate0/worldgen/tunnel_mesh_navigation_g001_v1.json \
  --output-dir results/manual_runs/navigation_replay_001/mesh
```

注意：使用全新的 `navigation_replay_001` 名称，`topology.json` 和 `mesh/` 必须不存在；工具会自动创建父目录。预期 parent 为 `tng_84998d00587e03dc`，mesh hash 为 `1e89fd07b834b1c494d4a320c4cc9d81c67af9a6a597462d3b40d8afcaf0656c`。
