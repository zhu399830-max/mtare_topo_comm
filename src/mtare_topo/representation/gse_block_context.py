"""Layout-correct compact frozen context pooling; no file/source authentication.

Caller must authenticate matching input, checkpoint and cache provenance before
calling. Tensor layout checks alone cannot detect a different observation's
otherwise valid cache. No nearest-coordinate matching or teacher IDs are used.
"""
import numpy as np
from .gse_block_points import BlockPoints


def bind_stamped_block_context(blocks,source_flat_ray_index,context,full_valid,
                               geometry_binding,cache_binding,*,chunk_size=2048):
    """Require matching provenance stamps; caller still verifies sealed bytes."""
    from .gse_candidate_context import ObservationBinding
    for stamp in (geometry_binding,cache_binding):
        if not isinstance(stamp,ObservationBinding):raise ValueError('typed observation binding required')
        stamp.validate()
    if geometry_binding!=cache_binding:raise ValueError('geometry/cache source disagreement')
    return pool_block_context(blocks,source_flat_ray_index,context,full_valid,chunk_size=chunk_size)


def pool_block_context(blocks,source_flat_ray_index,context,full_valid,*,chunk_size=2048):
    if not isinstance(blocks,BlockPoints) or not isinstance(chunk_size,int) or not 1<=chunk_size<=57600:
        raise ValueError('bounded block input/chunk required')
    index=np.asarray(source_flat_ray_index);memory=np.asarray(context);valid=np.asarray(full_valid)
    n=len(blocks.xyz_m);m=len(blocks.block_ids)
    if (index.shape!=(n,) or index.dtype.kind not in 'iu' or np.any(index<0) or np.any(index>=57600)
            or len(np.unique(index))!=n):
        raise ValueError('unique original source ray index per selected point required')
    if memory.shape!=(900,128) or memory.dtype!=np.float32 or not np.isfinite(memory).all():
        raise ValueError('finite compact900x128 float32 frozen context required')
    if valid.shape!=(57600,) or valid.dtype!=np.bool_:
        raise ValueError('full original bool validity mask required')
    if not np.all(valid[index]) or not np.array_equal(index//(16*720),blocks.frame_index):
        raise ValueError('selected ray validity/frame provenance mismatch')
    token=(index//(16*720))*180+(index%720)//4
    sums=np.zeros((m,128),np.float64);counts=np.zeros(m,np.int64)
    # Original-return weighting, including elevation rows sharing a token.
    # No all-population or full N x 128 high-dimensional cache is created.
    for start in range(0,n,chunk_size):
        stop=min(n,start+chunk_size);g=blocks.point_to_block[start:stop]
        np.add.at(sums,g,memory[token[start:stop]])
        np.add.at(counts,g,1)
    if m and np.any(counts==0):raise ValueError('empty declared block')
    pooled=(sums/counts[:,None]).astype(np.float32) if m else np.empty((0,128),np.float32)
    for a in (pooled,counts,token):a.flags.writeable=False
    return dict(context=pooled,point_count=counts,sensor_token_index=token)
