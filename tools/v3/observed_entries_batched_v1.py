"""Explicit scheduling adapter: preserve kernel and global ray identities.

Not installed in any teacher. Caller must bind the original kernel version.
"""
BATCH_RAYS=1152


def observed_entries_batched(kernel,caster,packed_rays,*,first_return,valid,
                             return_sources,center_m,radius_m=10.):
    # Empty requests still enter the original validator.
    if len(packed_rays)==0:
        return kernel(caster,packed_rays,first_return=first_return,valid=valid,
            return_sources=return_sources,center_m=center_m,radius_m=radius_m)
    if len(first_return)!=len(packed_rays) or len(valid)!=len(packed_rays) or len(return_sources)!=len(packed_rays):
        raise ValueError('complete aligned ray arrays required')
    entries=[]
    for offset in range(0,len(packed_rays),BATCH_RAYS):
        end=offset+BATCH_RAYS
        result=kernel(caster,packed_rays[offset:end],first_return=first_return[offset:end],
            valid=valid[offset:end],return_sources=return_sources[offset:end],
            center_m=center_m,radius_m=radius_m)
        if set(result)!={'entries','membership','physical_separation_certified'}:
            raise ValueError('bound original kernel schema required')
        if result['membership'] is not None or result['physical_separation_certified'] is not False:
            raise ValueError('unexpected geometry qualification')
        entries.extend(dict(e,ray_index=e['ray_index']+offset) for e in result['entries'])
    return dict(entries=entries,membership=None,physical_separation_certified=False)
