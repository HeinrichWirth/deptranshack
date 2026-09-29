"""Explain why an annotation seed is not a mathematically perfect seed."""
from common import *
from collections import Counter

def main():
    result={}
    for name in ('ORACLE_SEED_8','ORACLE_SEED_10','ORACLE_SEED_12','ORACLE_SEED_15','ORACLE_FRAME','ORACLE_SEED_FRAME','ORACLE_SEED_TEMPLATE8','ORACLE_SEED_TEMPLATE8_FRAME'):
        rows=[]
        for path in (OUT/'oracles'/name).glob('*/prediction.json'):
            p=load(path)
            if p['run'] not in SPLIT['validation']:continue
            first=p['steps'][0] if p['steps'] else {}
            rows.append(dict(status=p['reason'],accepted_steps=sum(s['status']=='ACCEPTED' for s in p['steps']),first_overlap_residual_p90=first.get('overlap_residual_p90')))
        result[name]=dict(starts=len(rows),stops=dict(Counter(r['status'] for r in rows)),no_accepted_march_steps=sum(r['accepted_steps']==0 for r in rows),
                          first_overlap_residual_p90=percentiles([r['first_overlap_residual_p90'] for r in rows if r['first_overlap_residual_p90'] is not None]))
    save(OUT/'audit/oracle_seed_profile_audit.json',result);print(result,flush=True)
if __name__=='__main__':main()
