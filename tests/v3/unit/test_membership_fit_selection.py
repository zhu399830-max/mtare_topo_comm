import copy
import pytest
from mtare_topo.data.gse_membership_fit_selection import select_membership_fit_windows


def fixture():
    positives, negatives, rechecks = [], [], []
    for parent in ('p0', 'p1', 'p2'):
        source = dict(split='fit', parent_id=parent, task=parent+'__variant',
                      source_sequence_id=1, frame_rows=[0, 1, 2, 3, 4])
        positives.append(dict(source=source, original_membership=True, terminal_node='end',
            opening_source='tube', evidence_path=parent+'.gz', evidence_sha256='a'*64))
        rechecks.append(dict(source=source, terminal_node='end', opening_source='tube',
            status='ORIGIN_PREREQUISITE_RETAINED', retained_witness_frames=[0,1],
            original_witness_frames=[0,1]))
        negatives.append(dict(source=dict(source, source_sequence_id=2, frame_rows=[5,6,7,8,9]),
            negative_memberships=1, evidence_file=parent+'neg.gz',
            storage={'compressed_sha256':'b'*64}))
    return positives, negatives, rechecks


def test_fixed_and_score_independent():
    p,n,r=fixture()
    expected=select_membership_fit_windows(p,n,r,parents=2)
    for row in p+n:
        row['model_score']=999
    assert select_membership_fit_windows(p[::-1],n[::-1],r[::-1],parents=2)==expected
    assert expected['windows']==4
    assert len({x['source']['parent_id'] for x in expected['records']})==2
    for a,b in zip(expected['records'][::2],expected['records'][1::2]):
        assert a['source']['task']==b['source']['task']
        assert a['source']['source_sequence_id']!=b['source']['source_sequence_id']


def test_no_padding_or_promotion():
    p,n,r=fixture()
    p[0]['original_membership']=None
    r[1]['retained_witness_frames']=[]
    with pytest.raises(ValueError,match='only 1'):
        select_membership_fit_windows(p,n,r,parents=2)


def test_no_cross_variant_substitution():
    p,n,r=fixture()
    n[0]['source']['task']='other_variant'
    with pytest.raises(ValueError,match='only 2'):
        select_membership_fit_windows(p,n,r,parents=3)


def test_conflicting_source_is_not_arbitrarily_selected():
    p,n,r=fixture()
    drift=copy.deepcopy(p[0]);drift['evidence_sha256']='c'*64
    with pytest.raises(ValueError,match='conflicting'):
        select_membership_fit_windows(p+[drift],n,r,parents=2)


def test_nonfit_not_used():
    p,n,r=fixture()
    for row in (p[0],n[0],r[0]):
        row['source']['split']='development'
    result=select_membership_fit_windows(p,n,r,parents=2)
    assert all(x['source']['parent_id']!='p0' for x in result['records'])
