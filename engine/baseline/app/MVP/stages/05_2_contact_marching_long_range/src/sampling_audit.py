"""Post-inference raw observation spacing of every >=100m research example."""
from long_common import *

def main():
    output=[]
    for e in load(OUT/'gallery/examples.json'):
        if (e.get('max_confirmed_observed_range') or 0)<100:continue
        folder=OUT/e['folder'];p=read_prediction(folder)
        with np.load(folder/'provenance.npz') as z:xyz=z['xyz'];source=z['source_frame']
        with np.load(folder/'point_evaluation.npz') as z:r=z['range'];d=z['distance'];v=z['evaluable']
        rows=[];confirmed=p['point_state']==2
        for lo in range(8,int(np.ceil(e['max_confirmed_observed_range']/4))*4,4):
            m=confirmed&(r>=lo)&(r<lo+4);good=m&v
            rows.append(dict(lo=lo,hi=lo+4,records=int(m.sum()),unique_xyz=len(np.unique(xyz[m],axis=0)),sources=len(np.unique(source[m])),matched5=float(np.mean(d[good]<=.05)) if good.any() else None,matched10=float(np.mean(d[good]<=.10)) if good.any() else None))
        rr=np.sort(np.unique(r[confirmed&(r>=8)]));maxgap=float(np.max(np.diff(rr))) if len(rr)>1 else None
        rec=dict(id=e['id'],max_radial_spacing_between_actual_observations=maxgap,fixed_4m_bins=rows,empty_4m_bins=sum(x['records']==0 for x in rows),under3_unique_bins=sum(x['unique_xyz']<3 for x in rows),scope='Post-hoc sampling audit only. Not used in selection, not a replacement for the frozen metric; radial spacing, not true along-track arc distance.')
        save(OUT/'100m_cases'/(e['id']+'_sampling.json'),rec);output.append(rec)
    save(OUT/'100m_cases/sampling_summary.json',output);print('SAMPLING AUDIT',len(output))

if __name__=='__main__':main()
