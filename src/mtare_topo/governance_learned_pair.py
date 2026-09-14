import hashlib
import json
from .governance import ValidationReport


def validate_card(card):
    errors=[];s=card.get('scope',{});a=card.get('approval',{})
    digest=hashlib.sha256(json.dumps(s,sort_keys=True).encode()).hexdigest()
    if card.get('schema_version')!='gse_learned_geometry_pair_audit_v1':errors.append('schema mismatch')
    if (s.get('worlds')!=['tunnel'] or s.get('raw_frames')!=294 or s.get('effective_five_frame_observations')!=290
            or s.get('independent_trajectories')!=1 or s.get('model_seed')!=0
            or s.get('existence_threshold')!=.5 or s.get('training_steps')!=0 or s.get('teacher_reads')!=0
            or s.get('protected_world_reads')!=0):errors.append('population or method drift')
    if (a.get('status')!='APPROVED' or a.get('scope_sha256')!=digest
            or a.get('authorized_operations')!=['audit'] or a.get('authorized_gates')!=[6]):errors.append('exact audit authority missing')
    windows=s.get('windows',[])
    if len(windows)!=290 or len({k for w in windows for k in w['source_frame_keys']})!=294:errors.append('window count mismatch')
    return ValidationReport(not errors,tuple(errors))
