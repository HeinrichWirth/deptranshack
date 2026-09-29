from . import ROOT,OUT
import long_common as lc
import time


def main():
    records=[]
    for name in ('results_performance_final2/freeze.json','results_performance_final/freeze.json','results_final_pipeline/FINAL_PIPELINE_FREEZE.json'):
        f=lc.load(ROOT/name)
        for field in ('files','code_files','dependency_files'):
            for path,expected in f.get(field,{}).items():
                if not path.startswith(('MVP/stages/','MVP/final_pipeline/','MVP/performance_final/','MVP/performance_final2/')):continue
                p=ROOT/path
                if isinstance(expected,dict):expected=expected.get('sha256')
                if not isinstance(expected,str) or len(expected)!=64:continue
                actual=lc.sha(p) if p.is_file() else None
                records.append(dict(freeze=name,file=path,expected=expected,actual=actual,unchanged=expected==actual))
    lc.save(OUT/'frozen_integrity.json',dict(time_ns=time.time_ns(),checked=len(records),unchanged=all(r['unchanged'] for r in records),rows=records))
    print('FROZEN_INTEGRITY',len(records),all(r['unchanged'] for r in records))

if __name__=='__main__':main()
