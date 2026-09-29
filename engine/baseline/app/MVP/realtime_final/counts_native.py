"""Non-numerical counters and zero-allocation disabled timers."""
from pathlib import Path
def main():
    d=Path(__file__).parent/'native';s=(d/'solver.cpp').read_text();h=(d/'trace.hpp').read_text()
    h=h.replace('Counters* target;std::string key;','Counters* target;const char* key;').replace('Timer(Counters* t,std::string k):target(t),key(std::move(k))','Timer(Counters* t,const char* k):target(t),key(k)')
    s=s.replace(' Pool pool;int cap,grid;',' size_t candidate_points=0,fit_points=0,max_candidate_points=0;\n Pool pool;int cap,grid;')
    s=s.replace('p.seeds={center};','candidate_points+=p.points.size();fit_points+=p.fit.size();max_candidate_points=std::max(max_candidate_points,p.points.size());p.seeds={center};')
    s=s.replace('d["python_native_calls"]=calls;','d["candidate_points"]=candidate_points;d["fit_points"]=fit_points;d["max_candidate_points"]=max_candidate_points;d["python_native_calls"]=calls;')
    m=(d/'march.hpp').read_text().replace('roi0=roi_points,broad0=broad_points;','roi0=roi_points,broad0=broad_points,cp0=solver.candidate_points,fp0=solver.fit_points;')
    m=m.replace('std::chrono::duration<double>(std::chrono::steady_clock::now()-step_start).count()});','std::chrono::duration<double>(std::chrono::steady_clock::now()-step_start).count(),double(solver.candidate_points-cp0),double(solver.fit_points-fp0)});')
    (d/'trace.hpp').write_text(h);(d/'solver.cpp').write_text(s);(d/'march.hpp').write_text(m)
if __name__=='__main__':main()
