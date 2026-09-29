"""Exact ROI AABB and shared immutable raw-frame index; no trajectory state."""
from pathlib import Path

def main():
    root=Path(__file__).parent/'native';s=(root/'solver.cpp').read_text();m=(root/'march.hpp').read_text()
    def rep(text,old,new):
        assert text.count(old)==1,(old[:80],text.count(old));return text.replace(old,new)
    store=r'''
struct FrameStore {
 std::map<int,std::shared_ptr<Cloud>> frames;std::mutex mutex;size_t builds=0;double build_seconds=0;
 void add_frame(int index,Arr xyz){auto a=xyz.unchecked<2>();if(a.shape(1)!=3)throw std::runtime_error("world Nx3 required");auto c=std::make_shared<Cloud>();c->frame=index;c->world.resize(a.shape(0));auto start=std::chrono::steady_clock::now();
  {py::gil_scoped_release release;for(py::ssize_t i=0;i<a.shape(0);++i)c->world[i]={a(i,0),a(i,1),a(i,2)};for(size_t i=0;i<c->world.size();++i)c->cells[cell(c->world[i])].push_back(int(i));std::lock_guard<std::mutex> lock(mutex);frames[index]=c;++builds;build_seconds+=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();while(!frames.empty()&&frames.begin()->first<index-11)frames.erase(frames.begin());}
 }
 std::map<int,std::shared_ptr<Cloud>> snapshot(const std::vector<int>&history){std::lock_guard<std::mutex> lock(mutex);std::map<int,std::shared_ptr<Cloud>> result;for(int k:history){auto it=frames.find(k);if(it==frames.end())throw std::runtime_error("snapshot frame expired/missing");result[k]=it->second;}return result;}
 py::dict stats(){std::lock_guard<std::mutex> lock(mutex);size_t n=0;for(auto&f:frames)n+=f.second->world.size();py::dict d;d["builds"]=builds;d["build_seconds"]=build_seconds;d["frames"]=frames.size();d["points"]=n;d["xyz_bytes"]=n*sizeof(X);return d;}
};
'''
    m=rep(m,'struct Obs{',store+'\nstruct Obs{')
    m=rep(m,'std::map<int,Cloud> clouds;','std::map<int,std::shared_ptr<Cloud>> clouds;bool tight_roi=false;')
    m=rep(m,'clouds[index]=std::move(c);','clouds[index]=std::make_shared<Cloud>(std::move(c));')
    m=rep(m,'auto&cloud=found->second;','auto&cloud=*found->second;')
    m=m.replace('clouds[index].world','clouds[index]->world')
    m=rep(m,' void diagnostics(bool trace,bool timers){',' void attach(FrameStore&store,const std::vector<int>&history){clouds=store.snapshot(history);}\n void options(bool exact_tight_roi){tight_roi=exact_tight_roi;}\n void diagnostics(bool trace,bool timers){')
    old='Cell lo=cell(sub(wc,{bound,bound,bound})),hi=cell(add(wc,{bound,bound,bound}));for(int index:history)'
    new=r'''X bounds={bound,bound,bound};
 if(tight_roi){
   // Every surviving point satisfies this exact slab/template box. Transform its
   // eight-corner AABB analytically; retain a guard and the original exact filters.
   double mv=f.oa[0]+(pack.lo[0]+pack.hi[0])/2,mw=f.oa[1]+(pack.lo[1]+pack.hi[1])/2;
   X mid=add(f.C,expand({4,mv,mw},f.B));wc=add(expand(mid,invR),translation);
   X half={4,(pack.hi[0]-pack.lo[0])/2+.14,(pack.hi[1]-pack.lo[1])/2+.14};bounds={1e-7,1e-7,1e-7};
   for(int j=0;j<3;++j){X axis=expand(f.B[j],invR);for(int d=0;d<3;++d)bounds[d]+=std::abs(axis[d])*half[j];}
 }
 Cell lo=cell(sub(wc,bounds)),hi=cell(add(wc,bounds));for(int index:history)'''
    m=rep(m,old,new)
    old='for(auto&old:p.blocks)child.blocks.push_back(std::make_shared<Block>(*old));'
    m=rep(m,old,'{Timer copy_timer(profile_context,"memory_allocation_copy");'+old+'}')
    s=rep(s,'.def("trace",&Marcher::export_trace);','.def("trace",&Marcher::export_trace).def("attach",&Marcher::attach).def("options",&Marcher::options);py::class_<FrameStore,std::shared_ptr<FrameStore>>(m,"FrameStore",py::module_local()).def(py::init<>()).def("add_frame",&FrameStore::add_frame).def("stats",&FrameStore::stats);')
    (root/'solver.cpp').write_text(s);(root/'march.hpp').write_text(m);print('EXACT_SHARED_ROI_IMPLEMENTED')

if __name__=='__main__':main()
