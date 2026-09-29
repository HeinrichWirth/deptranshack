#include <string>
#include <cstdint>
#include <numeric>
struct Event {std::string operation;int step,pack,candidate,iteration,trial;std::vector<double> values;std::vector<uint64_t> ids;};
struct Trace {bool enabled=false;int step=0,pack=-1;std::vector<Event> events;
 void add(std::string name,std::vector<double> values={},std::vector<uint64_t> ids={},int candidate=-1,int iteration=-1,int trial=-1){if(enabled)events.push_back({std::move(name),step,pack,candidate,iteration,trial,std::move(values),std::move(ids)});}
 py::list export_events(){py::list out;for(auto&e:events){py::dict d;d["operation"]=e.operation;d["step"]=e.step;d["pack"]=e.pack;d["candidate"]=e.candidate;d["iteration"]=e.iteration;d["trial"]=e.trial;d["values"]=e.values;d["ids"]=e.ids;out.append(d);}return out;}
};
thread_local Trace* trace_context=nullptr;
void event(std::string name,std::vector<double> values={},std::vector<uint64_t> ids={}){if(trace_context)trace_context->add(std::move(name),std::move(values),std::move(ids));}
using Counters=std::map<std::string,double>;
thread_local Counters* profile_context=nullptr;
struct Timer {Counters* target;const char* key;std::chrono::steady_clock::time_point start;
 Timer(Counters* t,const char* k):target(t),key(k){if(target)start=std::chrono::steady_clock::now();}
 ~Timer(){if(target)(*target)[key]+=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();}
};
struct OptEvent{std::string operation;int iteration,trial;std::vector<double> values;};
std::vector<double> flatten(const std::vector<std::array<double,2>>&p){std::vector<double> v;v.reserve(p.size()*2);for(auto&a:p)for(double x:a)v.push_back(x);return v;}
struct ScopeExit{std::function<void()> f;~ScopeExit(){f();}};
