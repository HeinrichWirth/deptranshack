"""Transport-only CDR1 PointCloud2 decoder; no ROS installation required."""
import struct
import numpy as np

FORMATS={1:'i1',2:'u1',3:'i2',4:'u2',5:'i4',6:'u4',7:'f4',8:'f8'}
def decode(blob):
    if blob[:2] not in (b'\x00\x01',b'\x00\x00'):raise ValueError('Only CDR1 supported')
    endian='<' if blob[1]==1 else '>';at=4
    def number(fmt,size):
        nonlocal at
        at=4+((at-4+size-1)//size)*size
        value=struct.unpack_from(endian+fmt,blob,at)[0];at+=size;return value
    def string():
        nonlocal at
        n=number('I',4)
        if n<1 or at+n>len(blob) or blob[at+n-1]!=0:raise ValueError('Invalid CDR string')
        s=blob[at:at+n-1].decode('utf-8');at+=n;return s
    sec=number('i',4);ns=number('I',4);frame=string()
    height=number('I',4);width=number('I',4);count=number('I',4);fields=[]
    for _ in range(count):fields.append((string(),number('I',4),number('B',1),number('I',4)))
    big=number('B',1);step=number('I',4);row=number('I',4);length=number('I',4)
    if row<width*step or length<height*row or at+length>=len(blob):raise ValueError('Truncated cloud')
    names=[];formats=[];offsets=[]
    for name,offset,typ,n in fields:
        if typ not in FORMATS or n<1:raise ValueError('Unsupported point field')
        base=np.dtype(('>' if big else '<')+FORMATS[typ])
        names.append(name);formats.append(base if n==1 else (base,(n,)));offsets.append(offset)
    dtype=np.dtype(dict(names=names,formats=formats,offsets=offsets,itemsize=step))
    points=np.ndarray((height,width),dtype=dtype,buffer=blob,offset=at,strides=(row,step)).reshape(-1)
    if not {'x','y','z'}.issubset(names):raise ValueError('XYZ missing')
    return dict(header_time_ns=sec*10**9+ns,frame_id=frame,height=height,width=width,point_step=step,row_step=row,is_dense=bool(blob[at+length]),fields=fields),points
