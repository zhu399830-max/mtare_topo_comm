from mtare_topo.topology.nonexclusive_relation_support import anchored_relation_support

def test_original_float32_boundaries():
    from mtare_topo.topology.nonexclusive_relation_support import LOW,HIGH
    r=anchored_relation_support([0,1,2],[0],[(0,1,HIGH),(0,2,LOW)])
    assert r.support_by_anchor==((0,((1,HIGH),)),)
    assert r.repulsion_by_anchor==((0,((2,LOW),)),)

def test_shared_surface_does_not_merge_anchors():
    r=anchored_relation_support([0,1,2],[0,1],[(0,2,.99),(1,2,.98),(0,1,.01)])
    assert r.anchors==(0,1) and r.shared_rays==(2,)
    assert r.support_by_anchor==((0,((2,.99),)),(1,((2,.98),)))
    assert not r.branch_identity_verified and not r.physical_connection_verified

def test_no_transitive_hallucinated_support():
    r=anchored_relation_support([0,1,2],[0],[(0,1,.99),(1,2,.99)])
    assert r.support_by_anchor==((0,((1,.99),)),)
    assert 2 in r.unsupported_rays

def test_unknown_and_absent_preserved():
    r=anchored_relation_support([0,1,2,3],[0],[(0,1,None),(0,2,.5)])
    assert r.unsupported_rays==(0,1,2,3)
    assert r.repulsion_by_anchor==((0,()),)

def test_order_does_not_change_result():
    pairs=[(0,2,.95),(1,2,.99),(0,1,.03)]
    assert anchored_relation_support([0,1,2],[0,1],pairs)==anchored_relation_support([2,1,0],[1,0],pairs[::-1])

def test_negative_never_deletes_an_observation():
    r=anchored_relation_support([0,1],[0],[(0,1,.01)])
    assert r.ray_ids==(0,1) and r.unsupported_rays==(0,1)
    assert r.repulsion_by_anchor==((0,((1,.01),)),)
