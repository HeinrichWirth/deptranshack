#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cmath>
#include <limits>
namespace py=pybind11;
using Array=py::array_t<double,py::array::forcecast>;
py::array_t<double> distances(Array points,Array anchor,Array templ) {
    if(points.ndim()!=2 || points.shape(1)!=2 || anchor.ndim()!=1 || anchor.shape(0)!=2 || templ.ndim()!=2 || templ.shape(1)!=2 || templ.shape(0)==0)
        throw py::value_error("Expected points Nx2, anchor 2, nonempty template Mx2");
    auto p=points.unchecked<2>();auto a=anchor.unchecked<1>();auto t=templ.unchecked<2>();
    py::array_t<double> output(points.shape(0));auto out=output.mutable_unchecked<1>();
    {py::gil_scoped_release release;
    for(py::ssize_t i=0;i<p.shape(0);++i) {
        const double x=p(i,0)-a(0),y=p(i,1)-a(1);
        double best=std::numeric_limits<double>::infinity();
        for(py::ssize_t j=0;j<t.shape(0);++j) {
            const double dx=x-t(j,0),dy=y-t(j,1);
            const double d=dx*dx+dy*dy;
            if(d<best)best=d;
        }
        out(i)=std::sqrt(best);
    }}
    return output;
}
PYBIND11_MODULE(_final_c4_native,m) {
    m.doc()="Exact float64 point-to-template distance spike; no solver changes";
    m.def("distances",&distances);
    m.attr("abi_version")=1;
}
