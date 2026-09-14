"""Version-specific archive binding; inherits bounded, fail-stop transport."""
from pathlib import Path
from historical_teacher_client_v1 import HistoricalTeacherClient


class V8HistoricalTeacherClient(HistoricalTeacherClient):
    worker_path=Path(__file__).with_name('v8_historical_teacher_worker_v1.py')
    archive_sha256='b963b3207c65b2ed69504f4d6040e9b94a05d18f9305dc35ca0ca2a3ff04efea'
