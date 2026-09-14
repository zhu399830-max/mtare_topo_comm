import hashlib,json
from mtare_topo.governance import ValidationReport
def validate_card(card):
    errors=[]
    try:
        s=card['scope'];a=card['approval']
        assert card['schema_version']=='gse_direction_task_pilot_inputs_v1'
        assert [b['parent'] for b in s['blocks']]==['S08_3d_loop_rich_C01','S04_3d_unicyclic_small_C06','S08_3d_loop_rich_C06']
        assert len(s['observations'])==12 and s['independent_units']==3 and s['labels_generated']==s['training_steps']==0
        for b in s['blocks']:
            rows=b['observations'];assert len(rows)==4 and all(r['split']=='fit' and r['task']==b['task'] for r in rows)
            assert all(x['frame_rows'][1:]==y['frame_rows'][:-1] for x,y in zip(rows,rows[1:]))
        assert len({(r['task'],f) for r in s['observations'] for f in r['frame_rows']})==24
        assert a['status']=='APPROVED' and a['authorized_operations']==['data_export'] and a['authorized_gates']==[3]
        assert a['scope_sha256']==hashlib.sha256(json.dumps(s,sort_keys=True).encode()).hexdigest()
    except (AssertionError,KeyError,TypeError):errors.append('direction task pilot exact population mismatch')
    return ValidationReport(not errors,tuple(errors))
