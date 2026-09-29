"""Copy the frozen solver and add diagnostics only. Never edits old stage."""
from . import ROOT
from pathlib import Path

def main():
    dest=Path(__file__).parent/'native';source=ROOT/'MVP/c4_v2/native'
    s=(source/'solver.cpp').read_text();m=(source/'march.hpp').read_text()
    def rep(text,old,new):
        assert text.count(old)==1,(old[:100],text.count(old))
        return text.replace(old,new)
    s=rep(s,'using Points=std::vector<V>;','using Points=std::vector<V>;\n#include "trace.hpp"')
    s=rep(s,'nn(V p,const Points&t){double best','nn(V p,const Points&t){Timer nn_timer(profile_context,"NN");double best')
    s=rep(s,'struct Candidate {V a;','struct Candidate {std::vector<OptEvent> trace;Counters profile;int order=0;double initial_cost=0;V a;')
    s=rep(s,'struct Pack {Points points,fit;','struct Pack {int id=0;Points points,fit;')
    s=rep(s,' Pool pool;int cap,grid;',' Trace* trace=nullptr;bool profiling=false;int next_pack=0;Counters profile;\n Pool pool;int cap,grid;')
    s=rep(s,'std::vector<double>&r){r.resize','std::vector<double>&r){Timer residual_timer(profile_context,"residual_evaluation");r.resize')
    s=rep(s,'V x){Candidate c;','V x){Timer lm_timer(profile_context,"LM_fitting");Candidate c;')
    s=rep(s,'lambda=.001;c.evals=1;','lambda=.001;c.evals=1;c.initial_cost=cost;bool tracing=trace&&trace->enabled;if(tracing){std::vector<double> v={x[0],x[1],cost};v.insert(v.end(),r.begin(),r.end());c.trace.push_back({"initial_residual",-1,-1,std::move(v)});}')
    s=rep(s,'residual(p.fit,t,y,q);++c.evals;auto&J','residual(p.fit,t,y,q);++c.evals;if(tracing){std::vector<double> v={y[0],y[1],h,dx};v.insert(v.end(),q.begin(),q.end());c.trace.push_back({"finite_difference",it,j,std::move(v)});}auto&J')
    s=rep(s,'   if(std::max(std::abs(g0)','   if(tracing)c.trace.push_back({"normal_equations",it,-1,{x[0],x[1],cost,A,B,D,g0,g1,lambda}});\n   if(std::max(std::abs(g0)')
    s=rep(s,'++c.evals;if(nc<cost){','++c.evals;if(tracing)c.trace.push_back({"trial",it,trial,{x[0],x[1],y[0],y[1],cost,nc,lambda,aa,dd,det,nc<cost?1.:0.}});if(nc<cost){')
    s=rep(s,'const Points&t){std::vector<double> inside;','const Points&t){Timer scoring_timer(profile_context,"candidate_scoring");std::vector<double> inside;')
    s=rep(s,'Pack p;p.center=center;','Timer generation_timer(profiling?&profile:nullptr,"candidate_generation");Pack p;p.id=next_pack++;p.center=center;')
    s=rep(s,'p.candidates.resize(p.seeds.size());return p;','p.candidates.resize(p.seeds.size());return p;')
    old='void execute(std::vector<Pack>&packs,const Points&t){std::vector<std::pair<size_t,size_t>> jobs;'
    new='void execute(std::vector<Pack>&packs,const Points&t){Timer batch_timer(profiling?&profile:nullptr,"candidate_batch_wall");if(trace&&trace->enabled)for(auto&p:packs){trace->pack=p.id;trace->add("pack",{p.center[0],p.center[1],p.gate,double(p.points.size()),double(p.fit.size()),double(p.seeds.size())});trace->add("candidate_points",flatten(p.points));trace->add("fit_points",flatten(p.fit));trace->add("initial_params",flatten(p.seeds));}std::vector<std::pair<size_t,size_t>> jobs;'
    s=rep(s,old,new)
    old='p.candidates[j]=optimize(p,t,p.seeds[j]);score(p.candidates[j],p,t);'
    new='Counters cp;auto* parent_profile=profile_context;profile_context=profiling?&cp:nullptr;p.candidates[j]=optimize(p,t,p.seeds[j]);score(p.candidates[j],p,t);p.candidates[j].order=int(j);p.candidates[j].profile=std::move(cp);profile_context=parent_profile;'
    s=rep(s,old,new)
    old='for(auto&p:packs){std::vector<Candidate> unique;'
    new='for(auto&p:packs){if(trace)trace->pack=p.id;for(auto&c:p.candidates){for(auto&v:c.profile)profile[v.first]+=v.second;if(trace&&trace->enabled){for(auto&e:c.trace)trace->add(e.operation,e.values,{},c.order,e.iteration,e.trial);trace->add("candidate_final",{c.a[0],c.a[1],c.initial_cost,c.cost,c.score,c.error,c.coverage,double(c.support),double(c.evals),double(c.iterations),c.success?1.:0.},{},c.order);}}std::vector<Candidate> unique;'
    s=rep(s,old,new)
    s=rep(s,'p.candidates=std::move(unique);','p.candidates=std::move(unique);if(trace&&trace->enabled){std::vector<double> v;for(auto&c:p.candidates){v.push_back(c.order);v.push_back(c.score);}trace->add("ranking",v);}')
    s=rep(s,'PYBIND11_MODULE(_c4_v2,m)','PYBIND11_MODULE(_c4_rt,m)')
    s=rep(s,'py::class_<Marcher>(m,"Marcher")','py::class_<Marcher>(m,"Marcher",py::module_local())')
    s=rep(s,'py::class_<Solver>(m,"Solver")','py::class_<Solver>(m,"Solver",py::module_local())')
    s=rep(s,'.def("stats",&Marcher::stats);','.def("stats",&Marcher::stats).def("diagnostics",&Marcher::diagnostics).def("trace",&Marcher::export_trace);')
    # No branch, acceptance, point membership, numerical solver rule changes here.
    m=rep(m,'X sensor(X world){return','X sensor(X world){Timer transform_timer(profile_context,"map_current_transform");return')
    m=rep(m,'FrameStep next_frame(const Path&p){auto obs','FrameStep next_frame(const Path&p){Timer frame_timer(profile_context,"PCA_frame_construction");auto obs')
    m=rep(m,'if(obs.empty())throw','if(trace_context&&trace_context->enabled){std::vector<uint64_t> ids;std::vector<double> v;for(auto&o:obs){ids.push_back(o.key);for(double x:o.xyz)v.push_back(x);}event("confirmed_current",v,ids);}if(obs.empty())throw')
    m=rep(m,' for(int it=0;it<30;++it){',' if(trace_context&&trace_context->enabled){std::vector<double> z={c[0],c[1],c[2]};for(auto&r:a)for(double x:r)z.push_back(x);event("pca_covariance",z);}\n for(int it=0;it<30;++it){')
    m=rep(m,'double pp=cs*cs*a[p][p]','event("pca_libm",{double(it),double(p),double(q),2*a[p][q],a[q][q]-a[p][p],angle,cs,sn});double pp=cs*cs*a[p][p]')
    m=rep(m,'return {unit(t),c};','event("pca_result",{t[0],t[1],t[2],a[k][k]});return {unit(t),c};')
    m=rep(m,'bool promote(Path&p,const Block&newblock){','bool promote(Path&p,const Block&newblock){Timer promotion_timer(profile_context,"tentative_confirmation");')
    m=rep(m,' Solver solver;std::map<int,Cloud> clouds;',' Trace trace_log;bool profiling=false;Counters profile;std::vector<std::vector<double>> step_profile;size_t broad_points=0,transformed_points=0;\n Solver solver;std::map<int,Cloud> clouds;')
    m=rep(m,' void add_frame(int index,',' void diagnostics(bool trace,bool timers){trace_log.enabled=trace;profiling=timers;solver.trace=&trace_log;solver.profiling=timers;}\n py::list export_trace(){return trace_log.export_events();}\n void add_frame(int index,')
    m=rep(m,'auto a=xyz.unchecked<2>();','Timer ingest_timer(profiling?&profile:nullptr,"history_ingest");auto a=xyz.unchecked<2>();')
    m=rep(m,'{py::gil_scoped_release release;for(size_t i=0;i<c.world.size();','{py::gil_scoped_release release;Timer index_timer(profiling?&profile:nullptr,"index_build");for(size_t i=0;i<c.world.size();')
    m=rep(m,'std::vector<int>ids;for(int x=','std::vector<int>ids;auto query_begin=std::chrono::steady_clock::now();for(int x=')
    m=rep(m,'std::sort(ids.begin(),ids.end());for(int i:ids){','std::sort(ids.begin(),ids.end());broad_points+=ids.size();if(profiling)profile["spatial_index_query"]+=std::chrono::duration<double>(std::chrono::steady_clock::now()-query_begin).count();Timer filtering_timer(profiling?&profile:nullptr,"ROI_transform_exact_filter");for(int i:ids){++transformed_points;')
    m=rep(m,'roi_points+=out.size();return out;','roi_points+=out.size();if(trace_log.enabled){std::vector<uint64_t> ids;std::vector<double> xyz;for(auto&o:out){ids.push_back(o.key);for(double x:o.uvw)xyz.push_back(x);}event("ROI_exact",xyz,ids);}return out;')
    m=rep(m,' if(!weak)continue;',' if(trace_log.enabled){std::vector<uint64_t> ids;for(auto&o:b->points)ids.push_back(o.key);trace_log.add("support_acceptance",{double(step),double(rank),c.a[0],c.a[1],double(N),shape,margin,second.p90,f.first,shift,weak?1.:0.},ids,c.order);}if(!weak)continue;')
    m=rep(m,'for(int k=1;k<=40;++k){++steps;','for(int k=1;k<=40;++k){trace_log.step=k;trace_log.pack=-1;auto step_start=std::chrono::steady_clock::now();auto fit0=solver.fits,eval0=solver.evaluations,roi0=roi_points,broad0=broad_points;ScopeExit step_timer{[&]{if(profiling)step_profile.push_back({double(k),double(roi_points-roi0),double(broad_points-broad0),double(solver.fits-fit0),double(solver.evaluations-eval0),std::chrono::duration<double>(std::chrono::steady_clock::now()-step_start).count()});}};++steps;')
    m=rep(m,'frames[i]=next_frame(active[i]);mapping','event("input_frame",{double(i),active[i].C[0],active[i].C[1],active[i].C[2],active[i].B[0][0],active[i].B[0][1],active[i].B[0][2],active[i].B[1][0],active[i].B[1][1],active[i].B[1][2],active[i].B[2][0],active[i].B[2][1],active[i].B[2][2]});frames[i]=next_frame(active[i]);mapping')
    m=rep(m,' if(branches.empty()){',' if(trace_log.enabled){std::vector<double> z;for(auto&b:branches){z.push_back(b.score);z.push_back(b.pending);z.push_back(b.blocks.back()->rank);}event("beam_before_ranking",z);}if(branches.empty()){')
    m=rep(m,'std::vector<Path>branches;','Timer beam_timer(profiling?&profile:nullptr,"beam_management_inclusive_promotion_copy");std::vector<Path>branches;')
    m=rep(m,'current=index;for(int h:history)','current=index;trace_log.events.clear();trace_log.step=0;trace_log.pack=-1;solver.next_pack=0;step_profile.clear();trace_context=trace_log.enabled?&trace_log:nullptr;profile_context=profiling?&profile:nullptr;ScopeExit reset_context{[]{trace_context=nullptr;profile_context=nullptr;}};for(int h:history)')
    m=rep(m,'py::dict out;py::list blocks;','Timer output_timer(profiling?&profile:nullptr,"output_materialization");event("winner",{path.score,path.search,double(path.blocks.size()),double(path.pending)});py::dict out;py::list blocks;')
    m=rep(m,'out["reason"]=path.reason;','out["reason"]=path.reason;if(trace_log.enabled)trace_log.add("stop:"+path.reason);')
    m=rep(m,'auto d=solver.stats();','auto d=solver.stats();d["profile"]=py::cast(profile);d["solver_profile"]=py::cast(solver.profile);d["step_profile"]=py::cast(step_profile);d["broad_points"]=broad_points;d["transformed_points"]=transformed_points;')
    (dest/'solver.cpp').write_text(s,encoding='utf-8');(dest/'march.hpp').write_text(m,encoding='utf-8')
    print('COPIED_AND_INSTRUMENTED',dest)

if __name__=='__main__':main()
