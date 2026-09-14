"""Generate a navigation index from the frozen backup manifest, not raw data."""
from pathlib import Path
from collections import defaultdict
import json

root = Path(__file__).resolve().parents[2]
doc = root / 'docs/github_evidence_20260914'
groups = defaultdict(list)
for line in (doc / 'included.jsonl').read_text().splitlines():
    row = json.loads(line)
    key = '/'.join(row['path'].split('/')[:3])
    groups[key].append(row)
lines = ['# 实验文件导航', '', '仅按原路径整理，不重新判定实验通过与否。图片和小文件在 Git，大文件按 included.jsonl 从 Release 恢复。', '', '| 原目录 | Git 文件 | 附件文件 | 图片示例 |', '|---|---:|---:|---|']
for key, rows in sorted(groups.items()):
    git = [r for r in rows if r['destination'] == 'git']
    pics = [r for r in git if Path(r['path']).suffix.lower() in {'.png', '.jpg', '.svg'}]
    pic = ('[查看](../../' + pics[0]['path'] + ')') if pics else '—'
    location = f'[{key}](../../{key})' if git else key
    lines.append(f'| {location} | {len(git)} | {len(rows)-len(git)} | {pic} |')
(doc / 'RUN_INDEX.md').write_text('\n'.join(lines) + '\n')
print(f'Indexed {len(groups)} original directories')
