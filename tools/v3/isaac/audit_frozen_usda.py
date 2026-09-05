#!/usr/bin/env python3
"""Parse one frozen USDA with the pinned OpenUSD runtime without resolving downloads."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from pxr import Sdf, Usd


def _string_list(values: Any) -> list[str]:
    return sorted({str(value) for value in values if str(value)})


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset", required=True, type=Path)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    asset = args.asset.resolve()
    output = args.output
    raw = asset.read_bytes()
    text_header = raw[:256].decode("utf-8", errors="replace")

    layer = Sdf.Layer.FindOrOpen(str(asset))
    if layer is None:
        raise RuntimeError(f"OpenUSD could not parse layer: {asset}")
    stage = Usd.Stage.Open(layer, Usd.Stage.LoadNone)
    if stage is None:
        raise RuntimeError(f"OpenUSD could not open stage: {asset}")

    sublayers = _string_list(layer.subLayerPaths)
    external_references = _string_list(layer.GetExternalReferences())
    composition_dependencies = _string_list(layer.GetCompositionAssetDependencies())
    external_asset_dependencies = _string_list(layer.GetExternalAssetDependencies())
    dependencies = sorted(
        set(sublayers)
        | set(external_references)
        | set(composition_dependencies)
        | set(external_asset_dependencies)
    )

    root_prims = [
        {"name": prim.name, "path": str(prim.path), "type_name": prim.typeName}
        for prim in layer.rootPrims
    ]
    traversed_prims = [
        {"path": str(prim.GetPath()), "type_name": prim.GetTypeName()}
        for prim in stage.Traverse()
    ]
    report = {
        "schema_version": "official_usda_dependency_audit_v1",
        "overall_status": "PASS_SELF_CONTAINED_USDA" if not dependencies else "FAIL_EXTERNAL_DEPENDENCIES",
        "openusd_version": list(Usd.GetVersion()),
        "asset_path": str(asset),
        "is_usda_text": text_header.lstrip().startswith("#usda"),
        "default_prim": layer.defaultPrim,
        "root_prims": root_prims,
        "traversed_prims": traversed_prims,
        "prim_count": len(traversed_prims),
        "sub_layer_paths": sublayers,
        "external_references": external_references,
        "composition_asset_dependencies": composition_dependencies,
        "external_asset_dependencies": external_asset_dependencies,
        "all_external_dependencies": dependencies,
        "external_dependency_count": len(dependencies),
        "claim_boundary": (
            "This report establishes only that one frozen USD layer parses and whether it "
            "declares external dependencies. It does not create a sensor, run Isaac, validate "
            "RTX output, or authorize project data use."
        ),
    }
    serialized = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if output == "-":
        print(serialized, end="")
    else:
        output_path = Path(output).resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(serialized, encoding="utf-8")
    return 0 if report["overall_status"].startswith("PASS") else 2


if __name__ == "__main__":
    raise SystemExit(main())
