#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <array>
#include <vector>
#include <unordered_map>
#include <cmath>
#include <algorithm>
#include <limits>
namespace py=pybind11;
using V=std::array<double,3>;using K=std::array<int,3>;
V add(V a,V b){return {a[0]+b[0],a[1]+b[1],a[2]+b[2]};}
V sub(V a,V b){return {a[0]-b[0],a[1]-b[1],a[2]-b[2]};}
V mul(V a,double x){return {a[0]*x,a[1]*x,a[2]*x};}
double dot(V a,V b){return a[0]*b[0]+a[1]*b[1]+a[2]*b[2];}
V cross(V a,V b){return {a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]};}
double norm(V a){return std::sqrt(dot(a,a));}
V unit(V a){double n=norm(a);if(n<1e-10)throw std::runtime_error("Degenerate envelope basis");return mul(a,1/n);}
struct KH {size_t operator()(const K& k)const{uint64_t h=1469598103934665603ULL;for(int v:k){h^=(uint32_t)v;h*=1099511628211ULL;}h^=h>>33;h*=0xff51afd7ed558ccdULL;return h^(h>>33);}};
K cell(V p){return {(int)std::floor(p[0]*2),(int)std::floor(p[1]*2),(int)std::floor(p[2]*2)};}
struct Hits {size_t count=0,extra=0,eligible=0;double range=std::numeric_limits<double>::infinity();int nearest=-1;std::vector<int> sample,all;
 void hit(int i,V p,bool extrap=false){++count;all.push_back(i);if(extrap)++extra;double r=norm(p);if(r<range){range=r;nearest=i;}if(sample.size()<256)sample.push_back(i);}
 py::dict result(){py::dict d;d["count"]=count;d["eligible_points"]=eligible;d["extrapolated_count"]=extra;d["nearest_index"]=nearest;d["nearest_range_m"]=nearest<0?py::none():py::cast(range);d["sample_indices"]=sample;d["all_indices"]=all;return d;}};
py::dict near(py::array_t<float,py::array::c_style|py::array::forcecast> points,V up,V forward,V right,double floor,double width,double height,double penetration,double distance){
 auto a=points.unchecked<2>();if(a.shape(1)<3)throw std::runtime_error("XYZ required");Hits hits;
 {py::gil_scoped_release release;for(int i=0;i<a.shape(0);++i){V p={a(i,0),a(i,1),a(i,2)};if(!std::isfinite(dot(p,p))||dot(p,p)==0)continue;
 ++hits.eligible;double u=dot(p,forward),v=dot(p,right),w=dot(p,up)-floor;
 if(u>=0&&u<=distance&&std::abs(v)<width/2-penetration&&w>0.100001&&w<height-penetration)hits.hit(i,p);}}
 return hits.result();}
struct Segment {V c,f,r,u;double length;bool extra;};
struct Envelope {
 std::vector<Segment> segments;std::unordered_map<K,std::vector<int>,KH> grid;V lo={1e30,1e30,1e30},hi={-1e30,-1e30,-1e30};double width,height,penetration;
 Envelope(py::array_t<double,py::array::c_style|py::array::forcecast> pair,double w,double h,double eps,int original):width(w),height(h),penetration(eps){
 auto p=pair.unchecked<3>();if(p.shape(0)<2||p.shape(1)!=2||p.shape(2)!=3)throw std::runtime_error("Matched rail pair required");
 auto v=[&](int i,int side){return V{p(i,side,0),p(i,side,1),p(i,side,2)};};
 for(int i=0;i<p.shape(0)-1;++i){V a=mul(add(v(i,0),v(i,1)),.5),b=mul(add(v(i+1,0),v(i+1,1)),.5);double length=norm(sub(b,a));
 if(!std::isfinite(length)||length<1e-8||length>5)throw std::runtime_error("Invalid/discontinuous rail segment");
 V f=unit(sub(b,a)),r=sub(v(i,1),v(i,0));r=unit(sub(r,mul(f,dot(r,f))));V u=unit(cross(f,r));if(u[2]<0)u=mul(u,-1);
 Segment s{a,f,r,u,length,i>=original-1};int id=segments.size();segments.push_back(s);
 V mn={1e30,1e30,1e30},mx={-1e30,-1e30,-1e30};
 for(int end=0;end<2;++end)for(int side:{-1,1})for(int top=0;top<2;++top){V q=add(add(add(a,mul(f,end*length)),mul(r,side*width/2)),mul(u,top*height));for(int j=0;j<3;++j){mn[j]=std::min(mn[j],q[j]);mx[j]=std::max(mx[j],q[j]);lo[j]=std::min(lo[j],q[j]);hi[j]=std::max(hi[j],q[j]);}}
 K first=cell(mn),last=cell(mx);for(int x=first[0];x<=last[0];++x)for(int y=first[1];y<=last[1];++y)for(int z=first[2];z<=last[2];++z)grid[{x,y,z}].push_back(id);
 }}
 py::dict query(py::array_t<float,py::array::c_style|py::array::forcecast> points,py::array_t<double,py::array::c_style|py::array::forcecast> pose,bool brute=false){
 auto a=points.unchecked<2>();auto P=pose.unchecked<2>();if(P.shape(0)!=4||P.shape(1)!=4||a.shape(1)<3)throw std::runtime_error("Bad query shape");Hits hits;
 {py::gil_scoped_release release;for(int i=0;i<a.shape(0);++i){V local={a(i,0),a(i,1),a(i,2)};if(!std::isfinite(dot(local,local))||dot(local,local)==0)continue;V world;
 ++hits.eligible;if(local[1]>0)continue;for(int j=0;j<3;++j)world[j]=P(j,0)*local[0]+P(j,1)*local[1]+P(j,2)*local[2]+P(j,3);
 if(world[0]<lo[0]||world[0]>hi[0]||world[1]<lo[1]||world[1]>hi[1]||world[2]<lo[2]||world[2]>hi[2])continue;
 auto test=[&](int k){auto &s=segments[k];V d=sub(world,s.c);double u=dot(d,s.f),v=dot(d,s.r),w=dot(d,s.u);return u>=0&&u<=s.length&&std::abs(v)<width/2-penetration&&w>0.100001&&w<height-penetration;};
 if(brute){for(int k=0;k<(int)segments.size();++k)if(test(k)){hits.hit(i,local,segments[k].extra);break;}}
 else{auto found=grid.find(cell(world));if(found==grid.end())continue;for(int k:found->second)if(test(k)){hits.hit(i,local,segments[k].extra);break;}}
 }}return hits.result();}
};
PYBIND11_MODULE(_obstacle_envelope,m){m.def("near",&near);py::class_<Envelope>(m,"Envelope").def(py::init<py::array_t<double,py::array::c_style|py::array::forcecast>,double,double,double,int>()).def("query",&Envelope::query,py::arg("points"),py::arg("pose"),py::arg("brute")=false);}
