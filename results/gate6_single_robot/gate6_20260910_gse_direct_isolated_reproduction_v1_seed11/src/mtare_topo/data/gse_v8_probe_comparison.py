"""Compare target content separately from non-model evidence prose."""


def counts(record):
    values = [v for row in record['membership'] for v in row]
    return dict(positive=sum(v is True for v in values), negative=sum(v is False for v in values),
                unknown=sum(v is None for v in values))


def compare(old, new):
    a,b=old['record'],new['record']
    if old['source_binding'] != new['source_binding']:
        raise ValueError('source binding changed')
    # All record fields other than anchors/membership must stay identical.
    if set(a) != set(b):
        raise ValueError('record schema changed')
    for name in set(a)-{'anchors','membership'}:
        if a[name] != b[name]:
            raise ValueError('unexpected '+name+' change')
    def geometry(anchor):
        return {k:v for k,v in anchor.items() if k != 'evidence'}
    for record in (a,b):
        if len(record['membership']) != len(record['openings']) or any(
                len(row) != len(record['anchors']) for row in record['membership']):
            raise ValueError('membership dimensions mismatch')
        if any(type(v) not in (bool,type(None)) for row in record['membership'] for v in row):
            raise ValueError('membership must be bool or unknown')
    changes=[]; evidence_changes=[]; used=set()
    for i,anchor in enumerate(a['anchors']):
        matches=[j for j,value in enumerate(b['anchors']) if geometry(value)==geometry(anchor)]
        if len(matches)!=1 or matches[0] in used:
            raise ValueError('old anchor geometry lost/changed/duplicated')
        j=matches[0];used.add(j)
        if anchor.get('evidence') != b['anchors'][j].get('evidence'):
            evidence_changes.append(dict(old_anchor=i,new_anchor=j,
                before=anchor.get('evidence'),after=b['anchors'][j].get('evidence')))
        for oi,row in enumerate(a['membership']):
            before,after=row[i],b['membership'][oi][j]
            if before is not None and after is not before:
                raise ValueError('old known membership changed')
            if before is not after:
                changes.append(dict(opening=oi,old_anchor=i,new_anchor=j,before=before,after=after))
    return dict(old_anchors=len(a['anchors']),new_anchors=len(b['anchors']),
        membership_changes=changes,evidence_text_changes=evidence_changes,
        old_counts=counts(a),new_counts=counts(b),complete_annotation=False)
