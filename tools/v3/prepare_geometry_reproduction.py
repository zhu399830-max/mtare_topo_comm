"""Restore the sealed development sources into a NEW isolated workspace.

Does not start Docker, create a material run, or change historical evidence.
"""
from _bootstrap import PROJECT_ROOT as ROOT
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile

SPEC = 'configs/v3/gate6/gse_branch_geometry_return_v1.json'
ARCHIVE = ROOT / 'docs/reproduction/gse_branch_geometry_return_v1'
TOOLS = ['_bootstrap.py', 'preflight.py', 'create_run.py']


def prepare(destination):
    destination = Path(destination).absolute()
    if destination.exists():
        raise ValueError('destination must not exist; historical workspaces are never reused')
    archive = ARCHIVE / 'frozen_sources.tar.gz'
    manifest = json.loads((ARCHIVE / 'archive_manifest.json').read_text())
    if hashlib.sha256(archive.read_bytes()).hexdigest() != manifest['sha256']:
        raise ValueError('archive hash mismatch')
    with tarfile.open(archive) as stream:
        members = stream.getmembers()
        names = set()
        for member in members:
            path = Path(member.name)
            if not member.isfile() or path.is_absolute() or '..' in path.parts or member.name in names:
                raise ValueError('unsafe or duplicate archive member')
            names.add(member.name)
        destination.mkdir(parents=True, exist_ok=False)
        for member in members:
            target = destination / member.name
            target.parent.mkdir(parents=True, exist_ok=True)
            with stream.extractfile(member) as source, target.open('xb') as output:
                shutil.copyfileobj(source, output)
    spec = json.loads((destination / SPEC).read_text())
    for name, digest in spec['source_sha256'].items():
        if hashlib.sha256((destination / name).read_bytes()).hexdigest() != digest:
            raise ValueError('restored source drift: ' + name)
    supplemental = {}
    for name in TOOLS:
        source = ROOT / 'tools/v3' / name
        target = destination / 'tools/v3' / name
        if target.exists():
            raise ValueError('supplement collides with frozen source')
        shutil.copyfile(source, target)
        supplemental[str(target.relative_to(destination))] = hashlib.sha256(target.read_bytes()).hexdigest()
    status = destination / 'results/project_status.json'
    status.parent.mkdir(exist_ok=True)
    shutil.copyfile(ROOT / 'results/project_status.json', status)
    supplemental[str(status.relative_to(destination))] = hashlib.sha256(status.read_bytes()).hexdigest()
    result = subprocess.run([sys.executable, 'tools/v3/preflight.py', '--spec', SPEC],
                            cwd=destination, capture_output=True, text=True)
    report = dict(archive_sha256=manifest['sha256'], restored_source_files=len(spec['source_sha256']),
                  supplemental_sha256=supplemental, preflight_exit_code=result.returncode,
                  preflight_stdout=result.stdout, preflight_stderr=result.stderr,
                  experiment_executed=False, image_included=False)
    (destination / 'RESTORE_REPORT.json').write_text(json.dumps(report, indent=2))
    if result.returncode:
        raise RuntimeError('isolated preflight failed; see RESTORE_REPORT.json')
    print(json.dumps(dict(workspace=str(destination), restored_source_files=report['restored_source_files'],
                          preflight_passed=True, experiment_executed=False)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--destination', required=True)
    parser.add_argument('--binding-fix', action='store_true', help='restore the observation-binding corrected version 8')
    parser.add_argument('--direct-support', action='store_true', help='restore direct-support return version 9')
    args = parser.parse_args()
    if args.binding_fix:
        SPEC = 'configs/v3/gate6/gse_branch_geometry_return_binding_v1.json'
        ARCHIVE = ROOT / 'docs/reproduction/gse_branch_geometry_return_binding_v1'
    if args.direct_support:
        if args.binding_fix:parser.error('select only one version')
        SPEC = 'configs/v3/gate6/gse_branch_geometry_direct_return_v1.json'
        ARCHIVE = ROOT / 'docs/reproduction/gse_branch_geometry_direct_return_v1'
    prepare(args.destination)
