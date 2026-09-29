from . import OUT
from .trace import replay,compact
from .compare_trace import first
import long_common as lc
import numpy as np

def difference(a,b):
    eq={k:np.array_equal(a[k],b[k],equal_nan=True) if a[k].dtype.kind not in 'US' else np.array_equal(a[k],b[k]) for k in a}
    return dict(ids_equal=eq['keys'],anchors_equal=eq['anchors'],all_equal=all(eq.values()),anchor_max_delta=float(np.max(abs(a['anchors']-b['anchors']))) if a['anchors'].shape==b['anchors'].shape else None,windows_reason=str(a['reason']),other_reason=str(b['reason']))

def main():
    rows=[];counter=[];dest=OUT/'input_diagnosis';dest.mkdir(exist_ok=True)
    manifest=lc.load(OUT/'captures/manifest.json')
    for case in manifest['rows']:
        a=OUT/'captures'/case['file'];b=OUT/'linux_inputs/captures'/case['file'];changes={};numeric={}
        with np.load(a) as x,np.load(b) as y:
            assert set(x.files)==set(y.files)
            for key in x.files:
                xx=x[key];yy=y[key]
                if np.array_equal(xx,yy,equal_nan=True):continue
                change=dict(run=case['run'],frame=case['frame'],field=key,shape_windows=list(xx.shape),shape_linux=list(yy.shape),
                    max_abs_delta=float(np.max(abs(xx-yy))) if xx.shape==yy.shape else None,changed_elements=int(np.sum(xx!=yy)) if xx.shape==yy.shape else None)
                rows.append(change);changes[key]=yy;numeric[key]=change
        output,trace,_,_=replay(a,1,True,False)
        actual,ltrace,_,_=replay(b,1,True,False)
        compact(dest/(case['file'][:-4]+'_windows_inputs.npz'),trace);compact(dest/(case['file'][:-4]+'_linux_inputs.npz'),ltrace)
        row=dict(run=case['run'],frame=case['frame'],changed_fields=list(changes),test='all Linux input fields, same Windows C++ solver',**difference(output,actual));counter.append(row)
        for key,value in changes.items():
            if key.startswith('world_'):continue
            out,_,_,_=replay(a,1,False,False,overrides={key:value});counter.append(dict(run=case['run'],frame=case['frame'],test='swap '+key+' only',**difference(output,out)))
        print('INPUT_DIAG',case['run'],case['frame'],list(changes),difference(output,actual),flush=True)
    lc.csv_write(OUT/'cross_os_input_difference.csv',rows);lc.csv_write(OUT/'cross_os_input_counterfactual.csv',counter)
    lc.save(OUT/'INPUT_DIAGNOSIS.json',dict(inputs=rows,counterfactual=counter))

if __name__=='__main__':main()
