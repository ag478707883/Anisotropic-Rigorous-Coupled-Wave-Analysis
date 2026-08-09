#pragma once

#include <cstddef>
#include <initializer_list>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include "rcwa/field.hpp"
#include "rcwa/material.hpp"
#include "rcwa/types.hpp"

namespace rcwaPy {

std::string lowercase(std::string text);
int normalizedWorkerCount(int requested, std::size_t jobCount);

pybind11::dtype makeDtype(std::initializer_list<std::pair<const char*, const char*>> fields);
pybind11::array makeStructuredArray(const pybind11::dtype& dtype, std::size_t rows);

rcwa::Tensor3 tensorFromArray(pybind11::handle obj);
rcwa::Material materialFromObject(pybind11::handle obj, std::string defaultName = "material");
rcwa::Material builtinMaterialModelFromObject(const std::string& name,
                                                  const std::string& model,
                                                  pybind11::handle parameters);

bool materialsIdenticalAtReferenceWavelength(const rcwa::Material& a,
                                                 const rcwa::Material& b);

rcwa::Polarization parsePolarization(std::string_view text);
const char* polarizationName(rcwa::Polarization pol);
rcwa::FieldComponent parseFieldComponent(std::string_view text);
rcwa::FieldPlane parseFieldPlane(std::string_view text);

} // namespace rcwa_py
