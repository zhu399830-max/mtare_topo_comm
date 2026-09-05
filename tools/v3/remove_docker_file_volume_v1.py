#!/usr/bin/env python3
"""Clone a Docker image while removing one invalid file-valued VOLUME entry."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
import subprocess
import tarfile
import tempfile


def canonical_json(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--destination", required=True)
    parser.add_argument("--remove-volume", required=True)
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="mtare_image_metadata_") as raw:
        root = Path(raw)
        source_tar = root / "source.tar"
        output_tar = root / "output.tar"
        with source_tar.open("wb") as stream:
            subprocess.run(["docker", "image", "save", args.source], stdout=stream, check=True)
        with tarfile.open(source_tar, "r") as archive:
            archive.extractall(root / "image", filter="data")
        image = root / "image"
        manifest_path = image / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if len(manifest) != 1:
            raise RuntimeError("expected exactly one image manifest")
        record = manifest[0]
        old_config = image / record["Config"]
        config = json.loads(old_config.read_text(encoding="utf-8"))
        volumes = config.get("config", {}).get("Volumes")
        if not isinstance(volumes, dict) or args.remove_volume not in volumes:
            raise RuntimeError("requested VOLUME entry is absent")
        del volumes[args.remove_volume]
        if not volumes:
            config["config"].pop("Volumes")
        config.setdefault("config", {}).setdefault("Labels", {})[
            "org.mtare.reproducibility.removed_invalid_volume"
        ] = args.remove_volume
        payload = canonical_json(config)
        config_name = hashlib.sha256(payload).hexdigest() + ".json"
        (image / config_name).write_bytes(payload)
        old_config.unlink()
        record["Config"] = config_name
        record["RepoTags"] = [args.destination]
        manifest_path.write_bytes(canonical_json(manifest))
        with tarfile.open(output_tar, "w") as archive:
            for path in sorted(image.rglob("*")):
                archive.add(path, arcname=path.relative_to(image), recursive=False)
        subprocess.run(["docker", "image", "load", "--input", str(output_tar)], check=True)
    image_id = subprocess.check_output(
        ["docker", "image", "inspect", args.destination, "--format", "{{.Id}}"], text=True
    ).strip()
    print(json.dumps({"destination": args.destination, "image_id": image_id, "removed_volume": args.remove_volume}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
