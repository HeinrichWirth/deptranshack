"""Exact first-row voxel selection using sorted packed integer keys."""
import numpy as np

def voxel_sample(xyz,size=.22):
    keys=np.floor(xyz/size).astype(np.int32)
    # Packing is exact only in this documented integer domain.
    if len(keys) and (keys.min() < -1048576 or keys.max() >=1048576):
        _,ids=np.unique(keys,axis=0,return_index=True)
    else:
        unsigned=keys.astype(np.int64)+1048576
        packed=(unsigned[:,0]<<42)|(unsigned[:,1]<<21)|unsigned[:,2]
        _,ids=np.unique(packed,return_index=True)
    return xyz[ids].astype(np.float64)
