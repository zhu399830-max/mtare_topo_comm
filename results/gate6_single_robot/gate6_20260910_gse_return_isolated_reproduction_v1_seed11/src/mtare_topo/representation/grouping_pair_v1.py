"""Only grouping changes: primitive packet or matched-M geometric FPS Voronoi."""
import numpy as np
from .gse_block_points import bind_block_points
from .gse_block_context import bind_stamped_block_context


def spatial_assignment(points, count):
    if (points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all()
            or not 1 <= count <= 4096 or len(points)>57600):
        raise ValueError('bounded observation and positive matched block count required')
    unique, inverse = np.unique(points.astype(np.float64),axis=0,return_inverse=True)
    if len(unique)<count:raise ValueError('cannot match M without coincident fake seeds')
    distance=np.full(len(unique),np.inf);assignment=np.zeros(len(unique),np.int64)
    seed=0; chosen=[]
    for group in range(count):
        chosen.append(seed)
        d=np.sum((unique-unique[seed])**2,axis=1)
        closer=d<distance # exact nearest ties retain earliest FPS seed
        assignment[closer]=group;distance=np.minimum(distance,d)
        distance[chosen]=0.
        seed=int(np.argmax(distance))
    result=assignment[inverse]
    if len(np.unique(result))!=count:raise ValueError('matched M not achieved')
    return result


def spatial_representation(compact, primitive):
    blocks=primitive['blocks']
    if not np.array_equal(blocks.xyz_m,compact.points_xyz_m):raise ValueError('point population drift')
    assignment=spatial_assignment(blocks.xyz_m,len(blocks.block_ids))
    grouped=bind_block_points(blocks.xyz_m,blocks.frame_index,assignment)
    context=bind_stamped_block_context(grouped,compact.source_flat_ray_index,
        compact.context,compact.full_valid,compact.binding,compact.binding)['context']
    return dict(blocks=grouped,context=context,binding=compact.binding)
