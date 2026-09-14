"""Range candidates retaining the detector's actual connected-column support."""
import numpy as np
from mtare_topo.semantics.range_exit_baseline import RangeExitBaseline
from mtare_topo.semantics.primitive_relation_nonlearning import ELEVATION_DEG


def sector_column_mask(sector, columns=720):
    start=sector['start_column'];end=sector['end_column']
    if type(start) is not int or type(end) is not int or not 0<=start<columns or not 0<=end<columns:raise ValueError('column bounds')
    count=(end-start)%columns+1
    if not np.isclose(sector['angular_width_deg'],count*360/columns,rtol=0,atol=1e-10):raise ValueError('sector width/endpoints conflict')
    result=np.zeros(columns,bool);result[(start+np.arange(count))%columns]=True
    return result


def range_proposals(ranges,valid,pose,key):
    ranges=np.asarray(ranges);valid=np.asarray(valid,bool);pose=np.asarray(pose)
    if ranges.shape!=(16,720) or valid.shape!=ranges.shape or pose.shape!=(4,4):raise ValueError('original scan and pose shapes required')
    if not np.isfinite(pose).all():raise ValueError('finite pose required')
    proposals=[]
    for sector in RangeExitBaseline().predict(ranges,valid,np.asarray(ELEVATION_DEG))['sectors']:
        mask=sector_column_mask(sector)
        selected=valid&(ranges>=4)&(np.abs(np.asarray(ELEVATION_DEG))<=5)[:,None]&mask[None,:]
        refs=[key+'/ray:'+str(int(i)) for i in np.flatnonzero(selected)]
        if not refs:continue
        a=np.deg2rad(sector['heading_robot_deg']);d=pose[:3,:3]@np.array([np.cos(a),np.sin(a),0.])
        proposals.append(dict(axis_start_xyz_m=pose[:3,3].tolist(),axis_target_xyz_m=(pose[:3,3]+4*d).tolist(),
            source_refs=refs,current_direction_supported=True,geometry_source_kind='range_sector_not_primitive',
            creates_edge=False,physical_opening=False,traversability='unknown',
            support_policy='exact_detector_connected_columns_v2',sector_columns=dict(start=sector['start_column'],end=sector['end_column'])))
    return proposals
