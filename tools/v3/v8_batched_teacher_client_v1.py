from pathlib import Path
from v8_historical_teacher_client_v1 import V8HistoricalTeacherClient


class V8BatchedTeacherClient(V8HistoricalTeacherClient):
    worker_path=Path(__file__).with_name('v8_batched_teacher_worker_v1.py')
