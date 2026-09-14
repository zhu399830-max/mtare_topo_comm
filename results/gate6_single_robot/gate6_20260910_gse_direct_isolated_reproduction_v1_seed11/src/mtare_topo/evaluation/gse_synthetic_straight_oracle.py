"""Analytic visible finite cap for the simple convex straight-tube fixtures.

No reference-target producer, fitted tolerance, source IDs, or hidden completion.
This is a synthetic positive-control oracle, not a general mine teacher.
"""
import numpy as np
from mtare_topo.data.cano_sensor_smoke import lidar_local_directions,world_directions


def visible_straight_cap_rays(case):
    if case['program']['type'] not in ('straight','terminal','visible_blocker'):
        raise ValueError('analytic oracle only covers the registered straight convex tube')
    first=case['program']['edges'][0];a,b=np.asarray(first['points'])
    if a[1:].any() or b[1:].any() or a[0]>=b[0]:raise ValueError('positive X straight fixture required')
    poses=np.asarray(case['poses_world_m']);origin=poses[-1]
    local=lidar_local_directions().reshape(-1,3).astype(np.float64)
    ay,az=case['half_axes_m'];power=case['shape_exponent'];result={}
    theta=np.arange(64)*2*np.pi/64
    polygon=np.stack((ay*np.sign(np.cos(theta))*np.abs(np.cos(theta))**(2/power),
                      az*np.sign(np.sin(theta))*np.abs(np.sin(theta))**(2/power)),axis=1)
    edge=np.roll(polygon,-1,axis=0)-polygon
    # Strictly inside every side of the actual convex 64-gon, not merely
    # inside the ideal superellipse outside the discretized wall.
    def in_section(yz):
        delta=yz[:,None,:]-polygon[None,:,:]
        return np.all(edge[None,:,0]*delta[:,:,1]-edge[None,:,1]*delta[:,:,0]>0,axis=1)
    for side,cap in enumerate((a,b)):
        supported=[]
        for slot,(pose,yaw) in enumerate(zip(poses,case['yaw_deg'],strict=True)):
            inside=in_section(pose[None,1:])[0] and a[0]<pose[0]<b[0]
            if not inside:raise ValueError('oracle origin must be strictly inside convex tube')
            d=world_directions(local,float(yaw));t=np.full(len(d),np.inf)
            np.divide(cap[0]-pose[0],d[:,0],out=t,where=d[:,0]!=0)
            finite=np.isfinite(t)&(t>0)
            ix=np.flatnonzero(finite);hit=pose+t[ix,None]*d[ix]
            section=in_section(hit[:,1:])
            roi=np.linalg.norm(hit-origin,axis=1)<10.
            supported.extend((slot*len(d)+ix[section&roi]).tolist())
        result[side]=supported
    return result
