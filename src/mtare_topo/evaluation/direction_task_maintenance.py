"""Score task assignment separately from model inputs and descriptor margins.

Reference candidate membership must be independently established by caller.
Missing membership remains unknown. Results apply only to covered references.
"""
def score_task_history(history,references):
    totals=dict(covered_observations=0,reference_tasks_not_recovered=0,duplicate_task_excess=0,
        conflicting_task_ids=0,known_candidate_outputs=0,unknown_candidate_outputs=0)
    rows=[];track_labels={}
    if len(history)!=len(references):raise ValueError('every observation requires explicit reference or unknown record')
    for event,ref in zip(history,references):
        if (event['segment'],event['order'])!=(ref['segment'],ref['order']):raise ValueError('reference order/segment mismatch')
        if not ref['evidence_ref']:raise ValueError('reference origin required, not model scores')
        labels=ref['candidate_labels'];visible=set(ref['observable_tasks']);groups={};unknown=0;known=0
        if any(v is not None and v not in visible for v in labels.values()):raise ValueError('candidate label outside supported reference')
        for a in event['assignments']:
            label=labels.get(a['candidate'])
            if label is None:unknown+=1;continue
            known+=1;groups.setdefault(label,set()).add(a['task_id'])
            track_labels.setdefault((event['segment'],a['task_id']),set()).add(label)
        missing=sorted(visible-set(groups));duplicate=sum(max(0,len(v)-1) for v in groups.values())
        # Track splits across time matter even if each frame has one prediction.
        rows.append(dict(segment=event['segment'],order=event['order'],reference_tasks_not_recovered=missing,
            duplicate_task_excess=duplicate,known_candidate_outputs=known,unknown_candidate_outputs=unknown))
        totals['covered_observations']+=int(bool(visible));totals['reference_tasks_not_recovered']+=len(missing)
        totals['duplicate_task_excess']+=duplicate;totals['known_candidate_outputs']+=known;totals['unknown_candidate_outputs']+=unknown
    label_tracks={}
    for (segment,track),labels in track_labels.items():
        for label in labels:label_tracks.setdefault((segment,label),set()).add(track)
    totals['fragmentation_excess']=sum(max(0,len(v)-1) for v in label_tracks.values())
    totals['conflicting_task_ids']=sum(len(v)>1 for v in track_labels.values())
    return dict(totals=totals,observations=rows,unknown_is_background=False,
        scope='covered_reference_task_assignment_only_not_physical_graph_correctness')
