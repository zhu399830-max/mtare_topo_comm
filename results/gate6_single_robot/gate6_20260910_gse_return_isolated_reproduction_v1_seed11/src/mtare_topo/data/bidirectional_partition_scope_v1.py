"""Missing-only833 paired partitions, preserving the existing27 packets."""
from collections import Counter
from .bidirectional_compact_scope_v1 import compile_scope as compact_scope

def compile_scope(root):
    base=compact_scope(root)
    missing=[r for r in base['observations'] if r['reused_partition'] is None]
    reused=[r for r in base['observations'] if r['reused_partition'] is not None]
    if len(missing)!=833 or len(reused)!=27:raise ValueError('fixed partition population drift')
    return dict(schema='bidirectional_partition_scope_v1',observations=missing,reused_observations=reused,
                metadata_sha256=base['metadata_sha256'],
                counts=dict(observations=833,reused=27,splits=dict(Counter(r['split'] for r in missing)),
                            unique_variant_frames=len({(r['source']['task'],f) for r in missing for f in r['source']['frame_rows']})),
                method='Unchanged fixed voxel and official native-SPG partition backends; exact same10m ROI rays for every representation; compact encoder caches reused.',
                fallback='Fail and seal; no point deletion, missing representation fallback or retry.',
                restrictions=base['limitations'],optimizer_steps=0,new_semantic_labels=0)
