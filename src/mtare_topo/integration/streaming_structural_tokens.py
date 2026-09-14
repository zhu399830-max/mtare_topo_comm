"""One resident full-SE3 extractor, bounded immutable per-request evidence.

The caller creates source-bound single-window manifests using received raw
buffers and full poses. This worker authenticates those bytes, never rebases an
epoch, publishes control, or claims a token was ready at scan acquisition.
Requests are sequential; overload must be recorded by the caller, not hidden
by replacing queued windows. Actual native-source/epoch binding is separate.
"""
from pathlib import Path
import time

from mtare_topo.integration import frozen_structural_token_worker as legacy
from mtare_topo.integration import frozen_structural_se3_worker as se3


class StreamingStructuralTokens:
    def __init__(self, root, output, *, device, max_requests, input_root=None, loader=None, clock=time.monotonic_ns):
        if type(max_requests) is not int or not 1 <= max_requests <= 20000:
            raise ValueError('EXPLICIT_BOUNDED_REQUEST_POPULATION')
        self.root = Path(root).resolve()
        self.input_root = self.root if input_root is None else Path(input_root).resolve()
        self.input_root.relative_to(self.root)
        self.output = Path(output).resolve()
        self.output.relative_to(self.root)
        self.output.mkdir(parents=True, exist_ok=False)
        self.clock, self.max_requests = clock, max_requests
        self.device, self.extractor = device, None
        self.loader = se3.load_registered_extractor if loader is None else loader
        self.requests, self.windows = set(), set()
        self.sequence_id = None
        self.last_frame = self.last_stamp = -1
        self.failed = False

    def preload(self):
        """Authenticate frozen model bytes once; no synthetic or real forward."""
        if self.extractor is None:
            self.extractor = self.loader(self.root, device=self.device)

    def process(self, request):
        started = self.clock()
        if self.failed:
            raise ValueError('STREAM_FATAL_NO_FURTHER_REQUESTS')
        if (type(request) is not dict or set(request) != {'schema_version','request_id','manifest'}
                or request['schema_version'] != 'streaming_structural_token_request_v1'):
            raise ValueError('STREAM_REQUEST_SCHEMA')
        request_id = request['request_id']
        if type(request_id) is not str or not request_id or len(request_id) > 256:
            raise ValueError('STREAM_REQUEST_ID')
        if request_id in self.requests:
            raise ValueError('DUPLICATE_REQUEST_NO_REPEAT_FORWARD')
        if len(self.requests) >= self.max_requests:
            raise ValueError('STREAM_REQUEST_CAP_REACHED')
        manifest_path = (self.root / request['manifest']['path']).resolve()
        if self.input_root not in manifest_path.parents:
            raise ValueError('STREAM_MANIFEST_OUTSIDE_INPUT_SCOPE')
        manifest = legacy._json_bytes(legacy._read_bound(self.root, request['manifest'], limit=65536))
        if (set(manifest) != {'schema_version','windows'}
                or manifest['schema_version'] != 'frozen_structural_token_windows_v1'
                or type(manifest['windows']) is not list or len(manifest['windows']) != 1):
            raise ValueError('EXACT_SINGLE_WINDOW_REQUEST_REQUIRED')
        entry = manifest['windows'][0]
        if self.input_root not in (self.root / entry['input']['path']).resolve().parents:
            raise ValueError('STREAM_PAYLOAD_OUTSIDE_INPUT_SCOPE')
        frames = entry.get('frames', [])
        if not isinstance(frames,list) or len(frames) != 5:
            raise ValueError('EXACT_FIVE_FRAMES_REQUIRED')
        frame, stamp = frames[-1].get('frame_order'), frames[-1].get('stamp_ns')
        if (type(frame) is not int or type(stamp) is not int
                or frame <= self.last_frame or stamp <= self.last_stamp):
            raise ValueError('STREAM_NONCAUSAL_CURRENT_FRAME')
        sequence_id, window_id = entry.get('sequence_id'), entry.get('window_id')
        if type(sequence_id) is not str or not sequence_id or (self.sequence_id is not None and sequence_id != self.sequence_id):
            raise ValueError('STREAM_SEGMENT_CHANGED')
        if type(window_id) is not str or not window_id or window_id in self.windows:
            raise ValueError('STREAM_WINDOW_ID')
        index = len(self.requests)
        self.requests.add(request_id); self.windows.add(window_id)
        self.sequence_id, self.last_frame, self.last_stamp = sequence_id, frame, stamp
        target = self.output / f'window_{index:06d}'
        def resident(root, *, device):
            self.preload()
            return self.extractor
        try:
            summary = legacy.run_worker(self.root, request['manifest'], target, device=self.device,
                _prepare=se3.prepare_window, _extractor_loader=resident, _preprocessing_version=se3.VERSION)
            completed = self.clock()
            record = legacy._json_bytes((target/'record_000000.json').read_bytes())
            response = dict(schema_version='streaming_structural_token_response_v1',request_id=request_id,
                window_id=window_id,sequence_id=sequence_id,source=record['source'],
                status=record['status'],reason=record.get('reason'),
                server_received_monotonic_ns=started,server_completed_monotonic_ns=completed,
                compute_and_evidence_wall_ns=completed-started,
                output_record=str(target/'record_000000.json'),
                output_record_sha256=legacy._sha((target/'record_000000.json').read_bytes()),
                checkpoint_sha256={key:value['sha256'] for key,value in summary['checkpoints'].items()},
                project_relative_output_record=str((target/'record_000000.json').relative_to(self.root)),
                token_ref=None,model_forward=record['model_forward'],training_steps=0,
                preprocessing_version=se3.VERSION,available_at_native_snapshot=False,
                availability_scope='must_check_client_receipt_and_exact_snapshot_separately',
                native_task_identity_verified=False,control_published=False)
            if record.get('token_ref'):
                response['token_ref'] = dict(path=str(target/record['token_ref']['path']),
                                            sha256=record['token_ref']['sha256'])
                response['project_relative_token_ref'] = dict(
                    path=str((target/record['token_ref']['path']).relative_to(self.root)),
                    sha256=record['token_ref']['sha256'])
            legacy._write_json(target/'stream_response.json',response)
            self.failed = summary['error'] is not None
            return response
        except Exception:
            self.failed = True
            raise
