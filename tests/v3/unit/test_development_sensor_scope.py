from copy import deepcopy
import pytest
from mtare_topo.data.development_sensor_scope import join_sensor_scope


def inputs():
    source = dict(task='S01_flat_tree_small_C01__ellipse', source_sequence_id=3, frame_rows=[0,1,2,3,4])
    provenance = dict(source, input_row=0, input_file_sha256='a'*64, coordinate_frame='current_sensor')
    compact = dict(observations=[dict(source=source, split='fit', feature_entry=dict(source=provenance),
                                      cache_path='cache', cache_sha256='b'*64)], manifests_sha256={})
    features = dict(tasks=[dict(task=source['task'], input_path='input', input_sha256='a'*64,
                    observation_count=1, observations=[dict(source, input_row=0, split='fit', parent_id='parent')])],
                    metadata_sha256={})
    return compact, features


@pytest.mark.parametrize('field,value', [('input_row',1), ('input_file_sha256','c'*64),
                                        ('coordinate_frame','world'), ('frame_rows',[1,2,3,4,5])])
def test_source_drift_rejected(field, value):
    compact, features = inputs()
    compact['observations'][0]['feature_entry']['source'][field] = value
    with pytest.raises(ValueError, match='source or split mismatch'):
        join_sensor_scope(compact, features)


def test_duplicate_sensor_and_population_drift():
    compact, features = inputs()
    with pytest.raises(ValueError, match='population drift'): join_sensor_scope(compact, features)
    features['tasks'][0]['observations'].append(deepcopy(features['tasks'][0]['observations'][0]))
    with pytest.raises(ValueError, match='duplicate sensor'): join_sensor_scope(compact, features)
