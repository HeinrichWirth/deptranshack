#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <array>
#include <vector>
#include <map>
#include <algorithm>
#include <cmath>
#include <limits>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <functional>
#include <chrono>
namespace py=pybind11;
using V=std::array<double,2>;
using Arr=py::array_t<double,py::array::forcecast>;
using Points=std::vector<V>;
Points read(Arr a){auto b=a.unchecked<2>();if(b.shape(1)!=2)throw std::runtime_error("expected Nx2");Points p(b.shape(0));for(py::ssize_t i=0;i<b.shape(0);++i){p[i]={b(i,0),b(i,1)};if(!std::isfinite(p[i][0])||!std::isfinite(p[i][1]))throw std::runtime_error("nonfinite coordinates");}return p;}
V readv(Arr a){auto b=a.unchecked<1>();if(b.shape(0)!=2)throw std::runtime_error("expected two parameters");return {b(0),b(1)};}
double sq(V a,V b){double x=a[0]-b[0],y=a[1]-b[1];return x*x+y*y;}
std::pair<double,int> nn(V p,const Points&t){double best=std::numeric_limits<double>::infinity();int ix=-1;for(size_t k=0;k<t.size();++k){double d=sq(p,t[k]);if(d<best){best=d;ix=int(k);}}return {std::sqrt(best),ix};}
// No Python callback, no GIL acquisition in worker threads. Deterministic slots.
class Pool {
 std::vector<std::thread> workers;std::mutex m;std::condition_variable ready,done;bool stop=false;size_t next=0,total=0,left=0;std::function<void(size_t)> job;
public:
 explicit Pool(int n){if(n<1||n>16)throw std::runtime_error("threads must be 1..16");if(n>1)for(int j=0;j<n;++j)workers.emplace_back([this]{for(;;){size_t k;{std::unique_lock<std::mutex> lock(m);ready.wait(lock,[&]{return stop||next<total;});if(stop)return;k=next++;}job(k);{std::lock_guard<std::mutex> lock(m);if(--left==0)done.notify_one();}}});}
 void run(size_t count,std::function<void(size_t)> fn){if(workers.empty()){for(size_t k=0;k<count;++k)fn(k);return;}{std::lock_guard<std::mutex> lock(m);job=std::move(fn);next=0;left=total=count;}ready.notify_all();std::unique_lock<std::mutex> lock(m);done.wait(lock,[&]{return left==0;});job=nullptr;}
 ~Pool(){{std::lock_guard<std::mutex> lock(m);stop=true;}ready.notify_all();for(auto &w:workers)w.join();}
};
struct Candidate {V a;double cost=0,score=-1,error=.5,coverage=0;int support=0,evals=0,iterations=0;bool success=false;};
struct Pack {Points points,fit;V center,lo,hi;double gate;bool fixed;int min_support;std::vector<V> seeds;std::vector<Candidate> candidates;};
struct Solver {
 Pool pool;int cap,grid;size_t calls=0,fits=0,evaluations=0,tasks=0;double fit_seconds=0;
 Solver(int threads,int iterations,int grid_size):pool(threads),cap(iterations),grid(grid_size){if(cap<1||cap>24)throw std::runtime_error("iteration cap 1..24");if(grid!=0&&grid!=5&&grid!=7&&grid!=9)throw std::runtime_error("grid 0/5/7/9");}
 static double residual(const Points&p,const Points&t,V a,std::vector<double>&r){r.resize(p.size());double cost=0;for(size_t k=0;k<p.size();++k){r[k]=nn({p[k][0]-a[0],p[k][1]-a[1]},t).first/.015;cost+=.5*std::log1p(r[k]*r[k]);}return cost;}
 Candidate optimize(const Pack&p,const Points&t,V x){Candidate c;c.a=x;if(p.fixed){c.success=true;return c;}std::vector<double> r,q,j0,j1;j0.resize(p.fit.size());j1.resize(p.fit.size());double cost=residual(p.fit,t,x,r),lambda=.001;c.evals=1;
  for(int it=0;it<cap;++it){c.iterations=it+1;
   for(int j=0;j<2;++j){double sign=x[j]>=0?1.:-1.,h=.001*sign*std::abs(x[j]);if(x[j]+h==x[j])h=std::sqrt(std::numeric_limits<double>::epsilon())*sign*std::max(1.,std::abs(x[j]));double lo=p.center[j]-p.gate,hi=p.center[j]+p.gate,dl=x[j]-lo,du=hi-x[j];if((x[j]+h<lo||x[j]+h>hi)&&std::abs(h)<=std::max(dl,du))h=-h;if(std::abs(h)>std::max(dl,du))h=du>=dl?du:-dl;V y=x;y[j]+=h;double dx=y[j]-x[j];residual(p.fit,t,y,q);++c.evals;auto&J=j==0?j0:j1;for(size_t k=0;k<r.size();++k)J[k]=(q[k]-r[k])/dx;}
   double A=0,B=0,D=0,g0=0,g1=0;for(size_t k=0;k<r.size();++k){double w=1/(1+r[k]*r[k]);A+=w*j0[k]*j0[k];B+=w*j0[k]*j1[k];D+=w*j1[k]*j1[k];g0+=w*j0[k]*r[k];g1+=w*j1[k]*r[k];}
   if(std::max(std::abs(g0),std::abs(g1))<1e-6){c.success=true;break;}bool accepted=false,finish=false;
   for(int trial=0;trial<8;++trial){double aa=A+lambda*std::max(A,1.),dd=D+lambda*std::max(D,1.),det=aa*dd-B*B;if(det<=0||!std::isfinite(det)){lambda*=10;continue;}V y={x[0]-(dd*g0-B*g1)/det,x[1]-(aa*g1-B*g0)/det};for(int j=0;j<2;++j)y[j]=std::clamp(y[j],p.center[j]-p.gate+1e-12,p.center[j]+p.gate-1e-12);double nc=residual(p.fit,t,y,q);++c.evals;if(nc<cost){double step=std::sqrt(sq(x,y)),gain=cost-nc;x=y;r.swap(q);cost=nc;lambda=std::max(lambda*.3,1e-9);accepted=true;finish=step<1e-6||gain<1e-7*std::max(1.,cost);break;}lambda*=10;}
   if(finish||!accepted){c.success=finish;break;}
  }c.a=x;c.cost=cost;return c;
 }
 static void score(Candidate&c,const Pack&p,const Points&t){std::vector<double> inside;for(V point:p.points){V q={point[0]-c.a[0],point[1]-c.a[1]};if(q[0]>=p.lo[0]&&q[0]<=p.hi[0]&&q[1]>=p.lo[1]&&q[1]<=p.hi[1]){double d=nn(q,t).first;inside.push_back(d);if(d<=.020)++c.support;}}if(!c.support)return;std::sort(inside.begin(),inside.end());size_t n=std::max(size_t(1),size_t(.7*inside.size()));double sum=0;for(size_t k=0;k<n;++k)sum+=inside[k];c.error=sum/n;int covered=0,total=0;for(size_t k=0;k<t.size();k+=2){++total;if(nn({t[k][0]+c.a[0],t[k][1]+c.a[1]},p.points).first<=.025)++covered;}c.coverage=double(covered)/total;c.score=std::exp(-c.error/.015)+.25*c.coverage+.1*(1-std::exp(-c.support/8.));}
 Pack prepare(Points pts,const Points&t,V center,double gate,bool fixed,int minimum){if(gate<=0||t.empty())throw std::runtime_error("positive gate and nonempty template required");Pack p;p.center=center;p.gate=gate;p.fixed=fixed;p.min_support=minimum;p.lo=t[0];p.hi=t[0];for(V a:t)for(int j=0;j<2;++j){p.lo[j]=std::min(p.lo[j],a[j]);p.hi[j]=std::max(p.hi[j],a[j]);}for(int j=0;j<2;++j){p.lo[j]-=.015;p.hi[j]+=.015;}
  for(V a:pts){V q={a[0]-center[0],a[1]-center[1]};if(q[0]>=p.lo[0]-gate&&q[0]<=p.hi[0]+gate&&q[1]>=p.lo[1]-gate&&q[1]<=p.hi[1]+gate)p.points.push_back(a);}std::sort(p.points.begin(),p.points.end());p.points.erase(std::unique(p.points.begin(),p.points.end()),p.points.end());if(p.points.size()<size_t(minimum))return p;
  std::map<std::array<int64_t,2>,V> cells;for(V a:p.points)cells.emplace(std::array<int64_t,2>{int64_t(std::floor(a[0]/.002)),int64_t(std::floor(a[1]/.002))},a);for(auto&cell:cells)p.fit.push_back(cell.second);
  p.seeds={center};if(!fixed){p.seeds.push_back({center[0]+gate*.7,center[1]});p.seeds.push_back({center[0]-gate*.7,center[1]});p.seeds.push_back({center[0],center[1]+gate*.7});p.seeds.push_back({center[0],center[1]-gate*.7});}
  if(grid&&!fixed){std::vector<std::pair<double,V>> ranked;std::vector<double> r;for(int i=0;i<grid;++i)for(int j=0;j<grid;++j){V a={center[0]+gate*(-1+2.*i/(grid-1)),center[1]+gate*(-1+2.*j/(grid-1))};ranked.push_back({residual(p.fit,t,a,r),a});++evaluations;}std::stable_sort(ranked.begin(),ranked.end(),[](auto&a,auto&b){return a.first<b.first;});p.seeds.clear();for(auto&entry:ranked){if(std::none_of(p.seeds.begin(),p.seeds.end(),[&](V a){return sq(a,entry.second)<.006*.006;}))p.seeds.push_back(entry.second);if(p.seeds.size()==5)break;}}
  p.candidates.resize(p.seeds.size());return p;
 }
 void execute(std::vector<Pack>&packs,const Points&t){std::vector<std::pair<size_t,size_t>> jobs;for(size_t i=0;i<packs.size();++i)for(size_t j=0;j<packs[i].seeds.size();++j)jobs.push_back({i,j});tasks+=jobs.size();pool.run(jobs.size(),[&](size_t k){auto[i,j]=jobs[k];auto&p=packs[i];p.candidates[j]=optimize(p,t,p.seeds[j]);score(p.candidates[j],p,t);});for(auto&p:packs){std::vector<Candidate> unique;for(auto&c:p.candidates){++fits;evaluations+=c.evals;if(std::none_of(unique.begin(),unique.end(),[&](const Candidate&o){return sq(c.a,o.a)<.006*.006;}))unique.push_back(c);}std::stable_sort(unique.begin(),unique.end(),[](auto&a,auto&b){return a.score>b.score;});p.candidates=std::move(unique);}}
 py::list batch(py::list input,Arr templ,int minimum){Points t=read(templ);struct In{Points p;V a;double gate;bool fixed;};std::vector<In> ins;for(auto item:input){auto row=py::cast<py::tuple>(item);ins.push_back({read(py::cast<Arr>(row[0])),readv(py::cast<Arr>(row[1])),py::cast<double>(row[2]),py::cast<bool>(row[3])});}std::vector<Pack> packs;auto clock=std::chrono::steady_clock::now();{py::gil_scoped_release release;++calls;for(auto&i:ins)packs.push_back(prepare(std::move(i.p),t,i.a,i.gate,i.fixed,minimum));execute(packs,t);fit_seconds+=std::chrono::duration<double>(std::chrono::steady_clock::now()-clock).count();}py::list out;for(auto&p:packs){py::list candidates;for(auto&c:p.candidates){py::dict d;d["anchor"]=py::cast(c.a);d["score"]=c.score;d["residual"]=c.error;d["support_unique"]=c.support;d["coverage"]=c.coverage;d["success"]=c.success;d["cost"]=c.cost;d["iterations"]=c.iterations;d["residual_evaluations"]=c.evals;candidates.append(d);}py::dict detail;detail["candidate_n"]=p.points.size();out.append(py::make_tuple(candidates,detail));}return out;}
 py::dict stats(){py::dict d;d["python_native_calls"]=calls;d["candidate_fits"]=fits;d["residual_evaluations"]=evaluations;d["native_tasks"]=tasks;d["native_fit_seconds"]=fit_seconds;return d;}
};
#include "march.hpp"
PYBIND11_MODULE(_c4_v2,m){py::class_<Marcher>(m,"Marcher").def(py::init<int,int,int>()).def("add_frame",&Marcher::add_frame).def("extend_track",&Marcher::extend).def("stats",&Marcher::stats);py::class_<Solver>(m,"Solver").def(py::init<int,int,int>(),py::arg("threads")=1,py::arg("iterations")=8,py::arg("grid")=0).def("fit_batch",&Solver::batch).def("stats",&Solver::stats);}

