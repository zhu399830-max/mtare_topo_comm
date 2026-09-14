from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'integration/native_structure_bridge'))
sys.path.insert(0, str(ROOT / 'tools/v3'))
from advice_sidecar_core import PinnedAdviceCache, reply_for_snapshot
from launch_native_structure_shadow import commands
from mtare_topo.integration.native_route_advice import make_route_advice


def snapshot():
    return dict(schema_version='native_route_snapshot_v1', epoch='session:4', stamp_ns=1000,
        max_age_ns=500, max_extra_cost_units=3, source_frame_keys=['registered_scan:900', 'registered_scan:1000'],
        candidate_ids=[1, 2], original_route=[0, 1, 2, 0], original_cost_units=30,
        native_edges=[[a, b, 0 if a == b else 10] for a in range(3) for b in range(3)])


class SidecarContractTests(unittest.TestCase):
    def test_no_cache_returns_exact_identity_with_reason(self):
        value = snapshot()
        original = deepcopy(value)
        result = reply_for_snapshot(value, now_ns=1100)
        self.assertEqual(result['reason'], 'NO_MODEL_CACHE')
        self.assertEqual(result['advice']['route'], value['original_route'])
        self.assertFalse(result['model_executed'])
        self.assertFalse(result['waypoint_published'])
        self.assertEqual(value, original)

    def test_stale_snapshot_is_not_replied_to(self):
        result = reply_for_snapshot(snapshot(), now_ns=1600)
        self.assertIsNone(result['advice'])
        self.assertEqual(result['decision']['reason'], 'ADVICE_NOT_FRESH')

    def cache(self, folder, advice, *, path='advice.json'):
        payload = json.dumps(advice).encode()
        (folder / 'advice.json').write_bytes(payload)
        manifest = dict(schema_version='native_route_advice_cache_v1', entries=[
            dict(epoch=advice['epoch'], path=path, sha256=hashlib.sha256(payload).hexdigest())])
        raw = json.dumps(manifest).encode()
        filename = folder / 'manifest.json'
        filename.write_bytes(raw)
        return filename, hashlib.sha256(raw).hexdigest()

    def test_pinned_cache_valid_reorder(self):
        value = snapshot()
        advice = make_route_advice(value, [0, 2, 1, 0], evidence_refs=['synthetic:registration'])
        with tempfile.TemporaryDirectory() as directory:
            cache = PinnedAdviceCache(*self.cache(Path(directory), advice))
            result = reply_for_snapshot(value, now_ns=1100, cache=cache)
            self.assertEqual(result['reason'], 'PINNED_EXACT_EPOCH_CACHE')
            self.assertTrue(result['decision']['changed'])

    def test_pinned_cache_wrong_epoch_not_rebased(self):
        value = snapshot()
        advice = make_route_advice(value, [0, 2, 1, 0], evidence_refs=['synthetic:registration'])
        advice['epoch'] = 'old_session:4'
        with tempfile.TemporaryDirectory() as directory:
            cache = PinnedAdviceCache(*self.cache(Path(directory), advice))
            result = reply_for_snapshot(value, now_ns=1100, cache=cache)
            self.assertEqual(result['reason'], 'EXACT_EPOCH_NOT_IN_CACHE')
            self.assertEqual(result['advice']['route'], value['original_route'])

    def test_wrong_sources_preserve_identity(self):
        value = snapshot()
        advice = make_route_advice(value, [0, 2, 1, 0], evidence_refs=['synthetic:registration'])
        advice['source_frame_keys'] = ['wrong']
        with tempfile.TemporaryDirectory() as directory:
            cache = PinnedAdviceCache(*self.cache(Path(directory), advice))
            result = reply_for_snapshot(value, now_ns=1100, cache=cache)
            self.assertEqual(result['reason'], 'CACHE_REJECTED:SOURCE_MISMATCH')
            self.assertEqual(result['advice']['route'], value['original_route'])

    def test_manifest_and_payload_tamper_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            advice = make_route_advice(snapshot(), [0, 1, 2, 0], evidence_refs=[])
            filename, digest = self.cache(path, advice)
            with self.assertRaisesRegex(ValueError, 'CACHE_MANIFEST_SHA_MISMATCH'):
                PinnedAdviceCache(filename, '0' * 64)
            (path / 'advice.json').write_text('{}')
            with self.assertRaisesRegex(ValueError, 'CACHE_ADVICE_SHA_MISMATCH'):
                PinnedAdviceCache(filename, digest)

    def test_path_escape_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            advice = make_route_advice(snapshot(), [0, 1, 2, 0], evidence_refs=[])
            with self.assertRaisesRegex(ValueError, 'CACHE_PATH_ESCAPE'):
                PinnedAdviceCache(*self.cache(Path(directory), advice, path='../outside.json'))

    def test_launcher_scope_and_single_native_node(self):
        planner, sidecar = commands(planner_seed=11, world='tunnel', output=Path('/tmp/synthetic-output'))
        self.assertEqual(planner[0], 'roslaunch')
        self.assertIn('planner_seed:=11', planner)
        self.assertIn('native_structure_advice_node.py', sidecar[1])
        with self.assertRaises(ValueError):
            commands(planner_seed=12, world='tunnel', output=Path('/tmp/synthetic-output'))
        root = ET.parse(ROOT / 'integration/native_structure_bridge/explore_native_shadow.launch').getroot()
        nodes = root.findall('node')
        self.assertEqual(len(nodes), 1)
        self.assertEqual(nodes[0].attrib['type'], 'tare_planner_node')
        params = {p.attrib['name']: p.attrib['value'] for p in nodes[0].findall('param')}
        self.assertEqual(params['native_structure_bridge_mode'], 'shadow')
        self.assertEqual(params['native_structure_wait_ms'], '100')

    def test_advice_node_has_only_advice_publisher(self):
        import ast
        tree = ast.parse((ROOT / 'tools/v3/native_structure_advice_node.py').read_text())
        publishers = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute) and node.func.attr == 'Publisher']
        self.assertEqual(len(publishers), 1)
        self.assertEqual(publishers[0].args[0].right.value, 'advice')

    def test_sidecar_callbacks_record_real_shaped_registry_without_ros(self):
        import native_structure_advice_node as node
        callbacks, published = {}, []
        value = snapshot()
        value.update(coordinate_frame='map', robot_position_xyz_m=[0., 0., 0.],
            candidate_positions_m={'1': [10., 0., 0.], '2': [20., 0., 0.]},
            candidate_records=[dict(id=cid, position_xyz_m=[10. * cid, 0., 0.],
                status=1, status_name='EXPLORING') for cid in (1, 2)])
        feedback = dict(schema_version='native_route_feedback_v1', session_id='session',
            epoch='session:4', stamp_ns=1100, source_frame_keys=value['source_frame_keys'],
            evidence_id='session:event:1', event_index=1, event='NATIVE_REGION_DISPATCH',
            candidate_id=1, position_xyz_m=[10., 0., 0.], waypoint_xyz_m=[9., 0., 0.],
            robot_position_xyz_m=[0., 0., 0.],
            dispatch_scope='published_native_waypoint_region_not_route_first_cell')
        fake = types.ModuleType('rospy')
        fake.Time = types.SimpleNamespace(now=lambda: types.SimpleNamespace(to_nsec=lambda: 1100))
        fake.init_node = lambda *args, **kwargs: None
        fake.Publisher = lambda *args, **kwargs: types.SimpleNamespace(publish=lambda msg: published.append(msg.data))
        def subscribe(topic, msgtype, callback, **kwargs):
            callbacks[topic.rsplit('/', 1)[-1]] = callback
            # A late callback during unregister must be ignored after close.
            return types.SimpleNamespace(unregister=lambda: callback(types.SimpleNamespace(data='{}')))
        fake.Subscriber = subscribe
        def spin():
            callbacks['candidates'](types.SimpleNamespace(data=json.dumps(value)))
            decision=dict(schema_version='native_route_decision_v1',epoch=value['epoch'],mode='shadow',
                changed=False,original_route=value['original_route'],route=value['original_route'],
                source_frame_keys=value['source_frame_keys'],native_candidates_preserved=True,native_edges_changed=False)
            callbacks['decision'](types.SimpleNamespace(data=json.dumps(decision)))
            callbacks['feedback'](types.SimpleNamespace(data=json.dumps(feedback)))
        fake.spin = spin
        fake.signal_shutdown = lambda reason: None
        fake.logerr = fake.logwarn = lambda *args: None
        std_msgs, messages = types.ModuleType('std_msgs'), types.ModuleType('std_msgs.msg')
        messages.String = lambda data: types.SimpleNamespace(data=data)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'sidecar'
            with patch.dict(sys.modules, {'rospy': fake, 'std_msgs': std_msgs, 'std_msgs.msg': messages}), \
                    patch.object(sys, 'argv', ['sidecar', '--output', str(output)]):
                self.assertEqual(node.main(), 0)
            registry = json.loads((output / 'native_region_tasks.json').read_text())
            self.assertEqual(len(registry['tasks']), 2)
            self.assertEqual(len(registry['attempts']), 1)
            self.assertEqual(len(registry['route_intents']), 1)
            self.assertEqual(registry['confirmed_traversals'], [])
            rows = [json.loads(row) for row in (output / 'trace.jsonl').read_text().splitlines()]
            self.assertEqual([row['trace_sequence'] for row in rows], [1, 2, 3])
            self.assertTrue(all(row['registry_decision']['accepted'] for row in rows))
            summary = json.loads((output / 'summary.json').read_text())
            self.assertEqual(summary['counts']['registry_exceptions'], 0)
        self.assertEqual(len(published), 1)
        self.assertEqual(json.loads(published[0])['route'], value['original_route'])


if __name__ == '__main__':
    unittest.main()
