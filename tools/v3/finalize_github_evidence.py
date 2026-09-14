"""Finish already-authorized transfers, verify, publish and commit receipts.

Does not overwrite releases, rewrite history, or touch research artifacts.
Pending state in project_status is a preparation snapshot; receipts supersede it.
"""
from pathlib import Path
import json
import os
import subprocess
import time

ROOT = Path(__file__).resolve().parents[2]
os.chdir(ROOT)
GH = '/home/zeng-workstation/.local/bin/gh'
REPO = 'zhu399830-max/mtare_topo_comm'
TAG = 'research-evidence-20260914'
SNAPSHOT = 'cbdca53e06741049442b24fe7745e39c8b6cdec0'
DOC = ROOT / 'docs/github_evidence_20260914'


def run(*args):
    return subprocess.check_output(args, text=True).strip()


def wait_process(pid, expected):
    start = time.monotonic()
    path = Path(f'/proc/{pid}/cmdline')
    while path.exists() and expected in path.read_bytes():
        if time.monotonic()-start > 43200:
            raise RuntimeError('Transfer exceeded 12h; no success recorded')
        time.sleep(10)


def main():
    assert run('git', 'rev-parse', 'HEAD') == SNAPSHOT, 'Unexpected branch movement'
    assert json.loads(run(GH, 'repo', 'view', REPO, '--json', 'visibility'))['visibility'] == 'PRIVATE'
    wait_process(254836, b'push')
    remote = run('git', 'ls-remote', 'origin', 'refs/heads/main').split()[0]
    if remote != SNAPSHOT:
        subprocess.run(['git', '-c', 'push.followTags=false', 'push', 'origin', 'main'], check=True)
    assert run('git', 'ls-remote', 'origin', 'refs/heads/main').split()[0] == SNAPSHOT
    print('Research snapshot remote SHA verified', flush=True)
    wait_process(256739, b'upload_github_evidence.py')
    # Re-running uploader reuses only assets with the exact server size and digest.
    for attempt in range(3):
        result = subprocess.run(['python3', 'tools/v3/upload_github_evidence.py'])
        if result.returncode == 0:
            break
        if attempt == 2:
            raise RuntimeError('Upload incomplete; preserve draft and partial assets')
        time.sleep(10)
    receipt = json.loads((DOC/'upload_receipt.json').read_text())
    assert receipt['state'] == 'REMOTE_RELEASE_ASSETS_SIZE_AND_SHA256_VERIFIED'
    subprocess.run([GH, 'release', 'edit', TAG, '--repo', REPO, '--target', SNAPSHOT,
                    '--draft=false', '--latest=false'], check=True)
    release = json.loads(run(GH, 'api', f'repos/{REPO}/releases/tags/{TAG}'))
    assert release['draft'] is False
    receipt.update(release_url=release['html_url'], published=True, verified_git_snapshot=SNAPSHOT)
    (DOC/'upload_receipt.json').write_text(json.dumps(receipt, ensure_ascii=False, indent=2)+'\n')
    status_path = ROOT/'results/project_status.json'
    status = json.loads(status_path.read_text())
    status['evidence_archive_20260914'].update(
        state='GIT_SNAPSHOT_AND_RELEASE_ASSETS_VERIFIED',
        verified_git_snapshot=SNAPSHOT, published_release=release['html_url'],
        release_asset_count=receipt['asset_count'])
    status['research_review_document_20260914']['current_document_remote_push_performed'] = True
    status_path.write_text(json.dumps(status, ensure_ascii=False, indent=2)+'\n')
    paths = ['docs/DECISION_LOG.md', 'docs/GITHUB_SYNC.md', 'docs/PROJECT_STATUS.md',
             'docs/github_evidence_20260914/README.md', 'docs/github_evidence_20260914/RUN_INDEX.md',
             'docs/github_evidence_20260914/upload_receipt.json', 'results/project_status.json',
             'tools/v3/upload_github_evidence.py', 'tools/v3/build_github_run_index.py',
             'tools/v3/finalize_github_evidence.py']
    assert not run('git', 'diff', '--cached', '--name-only'), 'Unrelated staged changes'
    assert run('git', 'rev-parse', 'HEAD') == SNAPSHOT, 'Unexpected branch movement'
    subprocess.run(['git', 'add', '--', *paths], check=True)
    subprocess.run(['git', '-c', 'gc.auto=0', 'commit', '-q', '-m',
                    'Verify and publish private research evidence archive'], check=True)
    subprocess.run(['git', '-c', 'push.followTags=false', 'push', 'origin', 'main'], check=True)
    head = run('git', 'rev-parse', 'HEAD')
    assert run('git', 'ls-remote', 'origin', 'refs/heads/main').split()[0] == head
    print(json.dumps({'state': 'GIT_AND_RELEASE_UPLOAD_VERIFIED', 'commit': head,
                      'release': release['html_url'], 'assets': receipt['asset_count']}), flush=True)


if __name__ == '__main__':
    main()
