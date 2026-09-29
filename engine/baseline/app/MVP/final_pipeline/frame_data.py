"""Immutable raw measurement input; annotation dimensions are not exposed."""
from dataclasses import dataclass
from types import MappingProxyType
import numpy as np
import laspy


@dataclass(frozen=True)
class FrameData:
    world: np.ndarray
    ring: np.ndarray
    point_index: np.ndarray
    measurements: object
    source_file: str
    frame: int
    source_bytes: int

    @staticmethod
    def read(path, frame):
        with laspy.open(path) as f:header=f.header
        if header.are_points_compressed:raise ValueError('Frozen RAW path supports uncompressed LAS only')
        raw=np.memmap(path,dtype=header.point_format.dtype(),mode='r',offset=header.offset_to_point_data,shape=(header.point_count,))
        world=np.column_stack([np.asarray(raw[a],dtype=np.float64)*header.scales[j]+header.offsets[j] for j,a in enumerate(('X','Y','Z'))])
        ring=np.asarray(raw['ring']).copy() if 'ring' in raw.dtype.names else np.full(len(world),-1,dtype=np.int16)
        point=np.asarray(raw['point_index']).copy() if 'point_index' in raw.dtype.names else np.arange(len(world),dtype=np.uint32)
        fields={name:np.asarray(raw[name]).copy() for name in ('intensity','intensity_raw','sensor_timestamp','frame_index','source_frame_index') if name in raw.dtype.names}
        for arr in [world,ring,point,*fields.values()]:arr.setflags(write=False)
        del raw
        return FrameData(world,ring,point,MappingProxyType(fields),str(path),frame,header.point_count*header.point_format.size)
