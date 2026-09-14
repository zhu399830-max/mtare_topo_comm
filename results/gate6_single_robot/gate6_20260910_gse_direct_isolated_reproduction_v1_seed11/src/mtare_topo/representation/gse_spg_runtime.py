"""File-level freeze for the isolated CPU partition runtime (no pip install)."""
import hashlib
import io
import platform
import sys
import sysconfig
from pathlib import Path


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f,'sha256').hexdigest()


def runtime_snapshot(root):
    import numpy as np
    import scipy
    from scipy.spatial import cKDTree
    import libcp
    import libply_c
    root=Path(root).resolve()
    prefix=root/'build/gse_spg_cpu_v1'
    if Path(sys.executable).resolve()!=Path('/usr/bin/python3.12').resolve():
        raise RuntimeError('isolated system Python3.12 required')
    for module in (np,scipy,libcp,libply_c):
        if not Path(module.__file__).resolve().is_relative_to(prefix):
            raise RuntimeError('unexpected native/package origin')
    if np.__version__!='1.26.4' or scipy.__version__!='1.11.4':
        raise RuntimeError('package version drift')
    # Exercise shared LAPACK, spatial and archive paths before enumerating maps.
    np.linalg.eigh(np.eye(3));cKDTree(np.eye(3)).query(np.zeros(3))
    np.savez_compressed(io.BytesIO(),x=np.eye(3))
    directories=[Path(np.__file__).parent,Path(scipy.__file__).parent,Path(sysconfig.get_path('stdlib'))]
    paths={Path(sys.executable).resolve(),Path(libcp.__file__).resolve(),Path(libply_c.__file__).resolve()}
    for directory in directories:
        for p in directory.rglob('*'):
            if ('__pycache__' not in p.parts and 'dist-packages' not in p.relative_to(directory).parts
                    and p.is_file() and (p.suffix in ('.py','.so') or '.so.' in p.name)):
                paths.add(p.resolve())
    for line in Path('/proc/self/maps').read_text().splitlines():
        tail=line.split(maxsplit=5)
        if len(tail)==6 and tail[-1].startswith('/'):
            p=Path(tail[-1])
            if p.is_file(): paths.add(p.resolve())
    def name(p):
        return str(p.relative_to(root)) if p.is_relative_to(root) else str(p)
    files={name(p):sha(p) for p in sorted(paths)}
    return dict(schema_version='gse_spg_runtime_files_v1',python=sys.version,numpy=np.__version__,
        scipy=scipy.__version__,machine=platform.machine(),files=files,
        scope='Python stdlib/package sources and extensions plus loaded native libraries; project source pinned separately',
        complete_runtime_freeze=True,system_installs=0)
