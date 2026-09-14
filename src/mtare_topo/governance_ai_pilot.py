"""Restricted non-blind AI annotation pilot, never a training qualification."""
import hashlib,json

def validate_fixed_fit_card(card):
    from mtare_topo.governance import ValidationReport
    errors=[];s=card.get('scope',{});a=card.get('approval',{});entries=s.get('entries',[])
    if (s.get('observations'),s.get('frames'),s.get('raw_ray_slots'))!=(12,60,691200):errors.append('exact12 five-frame review required')
    if len(entries)!=12 or len({e.get('parent') for e in entries})!=12:errors.append('twelve independent parents required')
    excluded=set(s.get('excluded_parents',[]))
    if not excluded:errors.append('old populations must be explicitly excluded')
    for e in entries:
        parent=e.get('parent','');frames=e.get('frame_rows',[])
        if e.get('split')!='fit' or parent.rsplit('_',1)[-1] not in {f'C{i:02}' for i in range(1,7)} or parent in excluded:errors.append('fit-only exclusion violation')
        if len(frames)!=5 or any(type(f)is not int for f in frames) or frames!=list(range(frames[0],frames[0]+5)):errors.append('five consecutive frames required')
        if not e.get('task','').startswith(parent+'__') or len(e.get('student_sha256',''))!=64:errors.append('source binding missing')
    for k,v in {'training_authorized':False,'unknown_is_background':False,'human_gold_count':0,'blind':False}.items():
        if s.get(k)!=v:errors.append('scope boundary '+k)
    if not s.get('spacing') or not s.get('bias') or not s.get('selection_rule'):errors.append('sampling and bias disclosure required')
    if a.get('status')!='APPROVED' or a.get('authorized_operations')!=['ai_annotation'] or a.get('authorized_gates')!=[3]:errors.append('exact annotation-only approval required')
    if not a.get('confirmation_reference') or a.get('scope_sha256')!=hashlib.sha256(json.dumps(s,sort_keys=True).encode()).hexdigest():errors.append('approval binding mismatch')
    return ValidationReport(not errors,tuple(errors),())


def validate_card(card):
    from mtare_topo.governance import ValidationReport
    errors=[];s=card.get('scope',{});a=card.get('approval',{})
    if s.get('observations')!=3 or s.get('frames')!=15 or s.get('raw_ray_slots')!=172800:
        errors.append('exact three five-frame observations required')
    entries=s.get('entries',[])
    expected={'S08_3d_loop_rich_C02__ellipse','S07_flat_loop_rich_C03__ellipse','S08_3d_loop_rich_C01__rounded_rectangle'}
    if len(entries)!=3 or {e.get('task') for e in entries}!=expected:errors.append('pilot population drift')
    for e in entries:
        if e.get('split')!='fit' or len(e.get('frame_rows',[]))!=5:errors.append('fit-only five-frame scope required')
        if len(e.get('student_sha256',''))!=64 or not e.get('student_path'):errors.append('student binding missing')
    for key,value in {'human_gold_count':0,'blind':False,'training_authorized':False,'unknown_is_background':False}.items():
        if s.get(key)!=value:errors.append('invalid pilot boundary '+key)
    if not s.get('spacing') or not s.get('quality_checks') or not s.get('images'):errors.append('sampling, evidence and quality plan required')
    if a.get('status')!='APPROVED' or a.get('authorized_gates')!=[3] or a.get('authorized_operations')!=['ai_annotation']:
        errors.append('exact user-approved Gate3 AI annotation required')
    if not a.get('confirmation_reference'):errors.append('user authorization reference missing')
    if a.get('scope_sha256')!=hashlib.sha256(json.dumps(s,sort_keys=True).encode()).hexdigest():errors.append('approval scope hash mismatch')
    return ValidationReport(not errors,tuple(errors),())
