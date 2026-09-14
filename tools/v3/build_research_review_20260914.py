"""Build an offline research retrospective from existing evidence, no experiments.

Only the explicitly referenced report figures are embedded. Dataset payloads,
weights, rendering engines and model code are never loaded. Historical evidence
is read-only; the output is a documentation artifact, not a scientific run.
"""
from pathlib import Path
import base64
import hashlib
import html
import json
import re

ROOT = Path(__file__).resolve().parents[2]
DOCS = ROOT / "docs"
NAME = "GSE_RESEARCH_REVIEW_20260914"
TEMPLATE = DOCS / "report_sources" / (NAME + ".template.html")


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def build():
    source = TEMPLATE.read_text(encoding="utf-8")
    figures = []

    def embed(match):
        relative, alt = match.groups()
        path = (ROOT / relative).resolve()
        path.relative_to(ROOT)
        raw = path.read_bytes()
        assert path.suffix == ".png" and raw.startswith(b"\x89PNG\r\n\x1a\n"), relative
        figures.append({"path": relative, "bytes": len(raw), "sha256": digest(raw)})
        return ('<img src="data:image/png;base64,' + base64.b64encode(raw).decode("ascii")
                + '" alt="' + alt + '" data-original="' + html.escape(relative, quote=True) + '">')

    output = re.sub(r'<img data-embed="([^"]+)" alt="([^"]+)">', embed, source)
    assert "data-embed=" not in output
    # Add provenance without altering original image pixels.
    iterator = iter(figures)
    def caption(match):
        fig = next(iterator)
        return match.group(1) + '<span class="figure-source">原图：' + html.escape(fig["path"]) + '</span></figcaption>'
    output = re.sub(r'(<figcaption>.*?)(?:</figcaption>)', caption, output, flags=re.S)
    links = []
    missing = []
    for href in sorted(set(re.findall(r'href="([^"]+)"', source))):
        if href.startswith(("#", "https:", "http:")) or href == NAME + ".manifest.json":
            continue
        path = (DOCS / href.split("#")[0]).resolve()
        path.relative_to(ROOT)
        if not path.is_file():
            missing.append(href)
            continue
        raw = path.read_bytes()
        links.append({"path": str(path.relative_to(ROOT)), "bytes": len(raw), "sha256": digest(raw)})
    if missing:
        raise ValueError("Missing report evidence links: " + repr(missing))
    raw = output.encode("utf-8")
    (DOCS / (NAME + ".html")).write_bytes(raw)
    manifest = {
        "kind": "documentation_retrospective_not_experiment",
        "as_of": "2026-09-14",
        "template": str(TEMPLATE.relative_to(ROOT)),
        "template_sha256": digest(TEMPLATE.read_bytes()),
        "html": NAME + ".html", "html_sha256": digest(raw), "html_bytes": len(raw),
        "embedded_figures": figures, "figure_count": len(figures), "reference_snapshots": links,
        "new_training_steps": 0, "new_simulations": 0, "historical_results_modified": False,
        "note": "Snapshot hashes identify documentation sources; not a fresh revalidation of all historical run seals. Historical protected-world outcomes are cited from existing reports, not reevaluated."
    }
    (DOCS / (NAME + ".manifest.json")).write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"html": str(DOCS / (NAME + ".html")), "bytes": len(raw), "figures": len(figures), "references": len(links), "missing_links": missing}, ensure_ascii=False))


if __name__ == "__main__":
    build()
