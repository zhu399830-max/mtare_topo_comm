from pathlib import Path
from v8_historical_teacher_client_v1 import V8HistoricalTeacherClient


class RequestScopedTeacherClient(V8HistoricalTeacherClient):
    worker_path=Path(__file__).with_name('request_scoped_teacher_worker_v1.py')
