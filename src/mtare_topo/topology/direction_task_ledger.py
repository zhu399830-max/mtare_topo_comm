"""Local direction-task bookkeeping, independent of the matching method.

No teacher input, descriptor threshold, >=3-direction trigger or physical
node/edge merge. A match carries provisional task identity, not traversability.
"""
from copy import deepcopy


class DirectionTaskLedger:
    def __init__(self):
        self.segment=None;self.order=None;self.previous={};self.tasks=[];self.history=[]

    def update(self, *, segment, order, candidates, previous_matches):
        if not isinstance(segment,str) or not segment or type(order) is not int:
            raise ValueError('explicit segment and integer causal order required')
        if self.order is not None and order<=self.order:raise ValueError('strict global observation order required')
        ids=[c['candidate'] for c in candidates]
        if any(type(i) is not int or i<0 for i in ids) or len(set(ids))!=len(ids):raise ValueError('unique local candidate indices required')
        if set(previous_matches)!=set(ids):raise ValueError('every candidate needs a match or explicit unknown')
        boundary=segment!=self.segment;previous={} if boundary else self.previous
        for c in candidates:
            if set(c)!={'candidate','source_frame_keys','target_xyz_m'}:raise ValueError('candidate schema excludes labels and hidden identity')
            if not c['source_frame_keys'] or len(set(c['source_frame_keys']))!=len(c['source_frame_keys']):raise ValueError('source keys required')
            import math
            if len(c['target_xyz_m'])!=3 or not all(math.isfinite(x) for x in c['target_xyz_m']):raise ValueError('finite target required')
        for matched in previous_matches.values():
            if matched is not None and (type(matched) is not int or matched not in previous):raise ValueError('match must reference preceding observation in same segment')
        # Validate everything before changing state.
        assignments=[];current={}
        for c in sorted(candidates,key=lambda c:c['candidate']):
            match=previous_matches[c['candidate']]
            if match is None:
                track=len(self.tasks)
                self.tasks.append(dict(task_id=track,segment=segment,state='provisional_unattempted',first_order=order,
                    physical_connection_verified=False,observations=[]))
            else:track=previous[match]
            self.tasks[track]['observations'].append(dict(order=order,candidate=c['candidate']))
            current[c['candidate']]=track
            assignments.append(dict(candidate=c['candidate'],task_id=track,matched_previous_candidate=match,
                association_verified=False,target_xyz_m=list(c['target_xyz_m'])))
        event=dict(segment=segment,order=order,assignments=assignments,created_task_ids=sorted({a['task_id'] for a in assignments if a['matched_previous_candidate'] is None}),
            physical_edges_created=0)
        self.history.append(deepcopy(event));self.previous=current;self.segment=segment;self.order=order
        return deepcopy(event)

    def snapshot(self):return deepcopy(dict(tasks=self.tasks,history=self.history,confirmed_edges=[],closed_loop=False))
