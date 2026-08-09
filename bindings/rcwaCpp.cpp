#include "rcwaSimulation.hpp"

#include <pybind11/pybind11.h>

PYBIND11_MODULE(rcwa_cpp, m) {
    m.doc() = "RETICOLO-aligned Python bindings for the RCWA C++ backend";
    bindRcwaCpp(m);
}
