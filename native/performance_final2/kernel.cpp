#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <vector>
#include <cmath>
#include <limits>
#include <algorithm>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <unordered_map>
namespace py=pybind11;
using ssize_t=py::ssize_t;
using Arr=py::array_t<double,py::array::forcecast>;
std::vector<double> copy(Arr a,int d) {
 auto b=a.request(); if(b.ndim!=2 || b.shape[1]!=d)throw std::runtime_error("wrong shape");
 std::vector<double> v(b.shape[0]*d);auto p=(char*)b.ptr;
 for(ssize_t i=0;i<b.shape[0];++i)for(int j=0;j<d;++j)v[i*d+j]=*(double*)(p+i*b.strides[0]+j*b.strides[1]);return v;
}
struct Problem {
 std::vector<double> p,t,tx,ty,f,scratch;double lo[2],hi[2],last[2];bool cached=false;bool soa=false;bool reuse_workspace=false;
 size_t fun_calls=0,jac_calls=0;
 Problem(Arr points,Arr templ,Arr lower,Arr upper,bool s):p(copy(points,2)),t(copy(templ,2)),soa(s){
  if(t.empty())throw std::runtime_error("empty template");auto l=lower.unchecked<1>(),h=upper.unchecked<1>();
  for(int j=0;j<2;++j){lo[j]=l(j);hi[j]=h(j);}for(size_t i=0;i<t.size();i+=2){tx.push_back(t[i]);ty.push_back(t[i+1]);}
 }
 void eval(const double *a,double *out,bool scaled=true){
  for(size_t i=0;i<p.size()/2;++i){double x=p[2*i]-a[0],y=p[2*i+1]-a[1],best=std::numeric_limits<double>::infinity();
   for(size_t k=0;k<t.size()/2;++k){double dx=x-(soa?tx[k]:t[2*k]),dy=y-(soa?ty[k]:t[2*k+1]);double q=dx*dx+dy*dy;if(q<best)best=q;}
   double v=std::sqrt(best);out[i]=scaled?v/.015:v;
  }
 }
 py::array_t<double> fun(Arr params,bool release=true){auto a=params.unchecked<1>();double x[2]={a(0),a(1)};py::array_t<double> out(p.size()/2);auto q=out.mutable_data();
  auto work=[&]{eval(x,q);f.assign(q,q+p.size()/2);last[0]=x[0];last[1]=x[1];cached=true;++fun_calls;};
  if(release){py::gil_scoped_release unlock;work();}else work();return out;
 }
 py::array_t<double> jac(Arr params,bool analytic=false){auto a=params.unchecked<1>();double x[2]={a(0),a(1)};ssize_t n=p.size()/2;
  py::array_t<double> out({n,ssize_t(2)},{ssize_t(sizeof(double)),ssize_t(n*sizeof(double))});double *q=out.mutable_data();
  py::gil_scoped_release unlock;++jac_calls;
  if(analytic){for(ssize_t i=0;i<n;++i){double best=std::numeric_limits<double>::infinity(),bx=0,by=0;
    for(size_t k=0;k<t.size()/2;++k){double dx=(p[2*i]-x[0])-t[2*k],dy=(p[2*i+1]-x[1])-t[2*k+1],v=dx*dx+dy*dy;if(v<best){best=v;bx=dx;by=dy;}}
    double d=std::sqrt(best);q[i]=d?-bx/d/.015:0;q[n+i]=d?-by/d/.015:0;}return out;}
  if(!cached || x[0]!=last[0] || x[1]!=last[1]){f.resize(n);eval(x,f.data());last[0]=x[0];last[1]=x[1];cached=true;}
  std::vector<double> temporary;if(reuse_workspace)scratch.resize(n);else temporary.resize(n);
  auto &shifted=reuse_workspace?scratch:temporary;
  for(int j=0;j<2;++j){double sign=x[j]>=0?1.:-1.;double h=.001*sign*std::abs(x[j]);
   if((x[j]+h)-x[j]==0)h=std::sqrt(std::numeric_limits<double>::epsilon())*sign*std::max(1.,std::abs(x[j]));
   double dl=x[j]-lo[j],du=hi[j]-x[j];bool violation=x[j]+h<lo[j] || x[j]+h>hi[j],fitting=std::abs(h)<=std::max(dl,du);
   if(violation && fitting)h*=-1.;if(!fitting)h=du>=dl?du:-dl;
   double xx[2]={x[0],x[1]};xx[j]=x[j]+h;double dx=xx[j]-x[j];eval(xx,shifted.data());
   for(ssize_t i=0;i<n;++i)q[j*n+i]=(shifted[i]-f[i])/dx;
  }return out;
 }
 size_t bytes()const{return (p.capacity()+t.capacity()+tx.capacity()+ty.capacity()+f.capacity()+scratch.capacity())*sizeof(double);}
};
// Isolated EXPERIMENTAL solver spike. Same residual/loss/bounds, different step
// and termination semantics from SciPy TRF. Never wired into production.
py::dict experimental_fit(Problem &p,Arr seed){auto a=seed.unchecked<1>();double x[2]={a(0),a(1)},lambda=1e-3;
 size_t n=p.p.size()/2;std::vector<double> f(n),trial(n),j0(n),j1(n);int nfev=1,iterations=0;double cost=0;
 {py::gil_scoped_release release;p.eval(x,f.data());for(double v:f)cost+=.5*std::log1p(v*v);
  while(nfev<24){++iterations;for(int j=0;j<2;++j){double sign=x[j]>=0?1.:-1.,h=.001*sign*std::abs(x[j]);
   if(x[j]+h==x[j])h=std::sqrt(std::numeric_limits<double>::epsilon())*sign*std::max(1.,std::abs(x[j]));
   double dl=x[j]-p.lo[j],du=p.hi[j]-x[j];if((x[j]+h<p.lo[j]||x[j]+h>p.hi[j])&&std::abs(h)<=std::max(dl,du))h=-h;
   if(std::abs(h)>std::max(dl,du))h=du>=dl?du:-dl;double xx[2]={x[0],x[1]};xx[j]+=h;double dx=xx[j]-x[j];
   p.eval(xx,trial.data());auto &jac=j?j1:j0;for(size_t i=0;i<n;++i)jac[i]=(trial[i]-f[i])/dx;}
   double A=lambda,B=0,D=lambda,g0=0,g1=0;for(size_t i=0;i<n;++i){double w=1/(1+f[i]*f[i]);A+=w*j0[i]*j0[i];B+=w*j0[i]*j1[i];D+=w*j1[i]*j1[i];g0+=w*j0[i]*f[i];g1+=w*j1[i]*f[i];}
   double det=A*D-B*B;double y[2]={std::clamp(x[0]-(D*g0-B*g1)/det,p.lo[0]+1e-12,p.hi[0]-1e-12),std::clamp(x[1]-(A*g1-B*g0)/det,p.lo[1]+1e-12,p.hi[1]-1e-12)};
   p.eval(y,trial.data());++nfev;double nc=0;for(double v:trial)nc+=.5*std::log1p(v*v);
   if(nc<cost){double step=std::hypot(y[0]-x[0],y[1]-x[1]);x[0]=y[0];x[1]=y[1];f.swap(trial);double gain=cost-nc;cost=nc;lambda=std::max(lambda*.3,1e-12);if(step<1e-8||gain<1e-8*cost)break;}else lambda*=10;
  }}
 py::dict out;out["x"]=py::make_tuple(x[0],x[1]);out["cost"]=cost;out["nfev"]=nfev;out["iterations"]=iterations;out["experimental"]=true;return out;
}
py::array_t<double> nn(Arr queries,Arr points){auto a=copy(queries,2),b=copy(points,2);if(b.empty())throw std::runtime_error("empty points");py::array_t<double> out(a.size()/2);double *o=out.mutable_data();py::gil_scoped_release unlock;
 for(size_t i=0;i<a.size()/2;++i){double best=std::numeric_limits<double>::infinity();for(size_t j=0;j<b.size()/2;++j){double x=a[2*i]-b[2*j],y=a[2*i+1]-b[2*j+1],v=x*x+y*y;if(v<best)best=v;}o[i]=std::sqrt(best);}return out;}
struct Pool {
 std::vector<std::thread> threads;std::mutex mutex;std::condition_variable ready,done;bool stop=false;size_t next=0,total=0,remaining=0;
 py::object fn=py::none();py::list inputs,outputs;std::string error;
 Pool(int n){if(n<1||n>64)throw std::runtime_error("workers 1..64");for(int j=0;j<n;++j)threads.emplace_back([this]{work();});}
 void work(){while(true){size_t i;{std::unique_lock<std::mutex> lock(mutex);ready.wait(lock,[&]{return stop||next<total;});if(stop)return;i=next++;}
   {py::gil_scoped_acquire gil;try{outputs[i]=fn(inputs[i]);}catch(py::error_already_set &e){std::lock_guard<std::mutex> lock(mutex);error=e.what();e.restore();PyErr_Clear();}}
   {std::lock_guard<std::mutex> lock(mutex);if(--remaining==0)done.notify_one();}}}
 py::list map(py::object callable,py::list jobs){fn=callable;inputs=jobs;outputs=py::list(jobs.size());error.clear();
  {std::lock_guard<std::mutex> lock(mutex);if(stop)throw std::runtime_error("pool closed");next=0;remaining=total=jobs.size();}
  {py::gil_scoped_release release;ready.notify_all();std::unique_lock<std::mutex> lock(mutex);done.wait(lock,[&]{return remaining==0;});}
  if(!error.empty())throw std::runtime_error(error);return outputs;}
 void close(){ {std::lock_guard<std::mutex> lock(mutex);stop=true;}ready.notify_all();for(auto &t:threads)if(t.joinable())t.join();}
 ~Pool(){close();}
};
struct Key{int x,y,z;bool operator==(const Key&o)const{return x==o.x&&y==o.y&&z==o.z;}};
struct Hash{size_t operator()(const Key&k)const{return size_t(k.x)*73856093u ^ size_t(k.y)*19349663u ^ size_t(k.z)*83492791u;}};
struct Grid{double cell;std::unordered_map<Key,std::vector<int>,Hash> cells;size_t n=0;
 Grid(Arr points,double c):cell(c){auto a=copy(points,3);n=a.size()/3;py::gil_scoped_release unlock;for(size_t i=0;i<n;++i)cells[{int(std::floor(a[3*i]/c)),int(std::floor(a[3*i+1]/c)),int(std::floor(a[3*i+2]/c))}].push_back(int(i));}
 py::array_t<int64_t> query(Arr center,double radius){auto c=center.unchecked<1>();int lo[3],hi[3];for(int j=0;j<3;++j){lo[j]=int(std::floor((c(j)-radius)/cell));hi[j]=int(std::floor((c(j)+radius)/cell));}std::vector<int64_t> ids;
  {py::gil_scoped_release unlock;for(int x=lo[0];x<=hi[0];++x)for(int y=lo[1];y<=hi[1];++y)for(int z=lo[2];z<=hi[2];++z){auto it=cells.find({x,y,z});if(it!=cells.end())ids.insert(ids.end(),it->second.begin(),it->second.end());}std::sort(ids.begin(),ids.end());}
  py::array_t<int64_t> out(ids.size());std::copy(ids.begin(),ids.end(),out.mutable_data());return out;}
 size_t bytes()const{size_t b=0;for(auto &x:cells)b+=sizeof(Key)+sizeof(std::vector<int>)+x.second.capacity()*sizeof(int);return b;}
};
PYBIND11_MODULE(_c4_sprint,m){m.def("exact_nn_bruteforce",&nn);
 py::class_<Problem>(m,"Problem").def(py::init<Arr,Arr,Arr,Arr,bool>(),py::arg("points"),py::arg("template"),py::arg("lower"),py::arg("upper"),py::arg("soa")=false)
 .def("fun",&Problem::fun,py::arg("params"),py::arg("release")=true).def("jac",&Problem::jac,py::arg("params"),py::arg("analytic")=false).def("bytes",&Problem::bytes)
 .def_readonly("fun_calls",&Problem::fun_calls).def_readonly("jac_calls",&Problem::jac_calls).def_readwrite("reuse_workspace",&Problem::reuse_workspace);
 m.def("experimental_fit",&experimental_fit);
 py::class_<Pool>(m,"Pool").def(py::init<int>()).def("map",&Pool::map).def("close",[](Pool &p){py::gil_scoped_release r;p.close();});
 m.def("native_evaluate_candidate_batch",[](py::object f,py::list jobs){py::list out;for(auto j:jobs)out.append(f(j));return out;});
 py::class_<Grid>(m,"Grid").def(py::init<Arr,double>()).def("query",&Grid::query).def("bytes",&Grid::bytes);
}
