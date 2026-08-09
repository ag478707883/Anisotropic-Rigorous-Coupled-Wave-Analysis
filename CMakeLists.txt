cmake_minimum_required(VERSION 3.20)
project(RCWA_Anisotropic LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(RCWA_ENABLE_PYTHON_BINDINGS "Build Python extension bindings" ON)
option(RCWA_SINGLE_PRECISION "Build the solver with float/complex-float arithmetic" OFF)
option(RCWA_ENABLE_IPO "Enable interprocedural optimization for release builds when available" ON)
set(RCWA_REQUIRE_FFTW ON)

include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/deps.cmake)

rcwa_add_mkl()
rcwa_add_fftw()
rcwa_add_catch2()

# Library sources

set(RCWA_CORE_SOURCES
    src/rcwa/fourier.cpp
    src/rcwa/li_factorization.cpp
    src/rcwa/material.cpp
    src/rcwa/matrix.cpp
    src/rcwa/physics.cpp
    src/rcwa/wavevector.cpp
)

set(RCWA_SOLVER_SOURCES
    src/rcwa/field.cpp
    src/rcwa/geometry.cpp
    src/rcwa/engine.cpp
    src/rcwa/layer.cpp
    src/rcwa/pattern.cpp
    src/rcwa/solver.cpp
    src/rcwa/smatrix.cpp
    src/rcwa/spectrum.cpp
    src/rcwa/results.cpp
    src/rcwa/scalar_stack.cpp
)

set(RCWA_SOURCES
    ${RCWA_CORE_SOURCES}
    ${RCWA_SOLVER_SOURCES}
)

add_library(rcwa STATIC ${RCWA_SOURCES})
target_include_directories(rcwa PUBLIC include)
set_target_properties(rcwa PROPERTIES POSITION_INDEPENDENT_CODE ON)

set(_rcwa_ipo_supported OFF)
if(RCWA_ENABLE_IPO)
    include(CheckIPOSupported)
    check_ipo_supported(RESULT _rcwa_ipo_supported OUTPUT _rcwa_ipo_error)
    if(NOT _rcwa_ipo_supported)
        message(STATUS "RCWA: IPO/LTO unavailable: ${_rcwa_ipo_error}")
    endif()
endif()

function(rcwa_enable_release_ipo target_name)
    if(_rcwa_ipo_supported)
        set_property(TARGET ${target_name} PROPERTY INTERPROCEDURAL_OPTIMIZATION_RELEASE TRUE)
        set_property(TARGET ${target_name} PROPERTY INTERPROCEDURAL_OPTIMIZATION_RELWITHDEBINFO TRUE)
        set_property(TARGET ${target_name} PROPERTY INTERPROCEDURAL_OPTIMIZATION_MINSIZEREL TRUE)
    endif()
endfunction()

rcwa_enable_release_ipo(rcwa)

function(rcwa_collect_windows_runtime_dirs out_var)
    set(_runtime_dirs)
    if(WIN32)
        if(MKL_INCLUDE_DIR)
            get_filename_component(_rcwa_mkl_root "${MKL_INCLUDE_DIR}" DIRECTORY)
            list(APPEND _runtime_dirs "${_rcwa_mkl_root}/bin")
        endif()

        foreach(_rcwa_compiler_bin
                "$ENV{ONEAPI_ROOT}/compiler/latest/bin"
                "C:/Program Files (x86)/Intel/oneAPI/compiler/latest/bin"
                "C:/Program Files/Intel/oneAPI/compiler/latest/bin")
            if(EXISTS "${_rcwa_compiler_bin}")
                list(APPEND _runtime_dirs "${_rcwa_compiler_bin}")
            endif()
        endforeach()

        if(CMAKE_CXX_COMPILER)
            get_filename_component(_rcwa_cxx_runtime_dir "${CMAKE_CXX_COMPILER}" DIRECTORY)
            list(APPEND _runtime_dirs "${_rcwa_cxx_runtime_dir}")
        endif()

        list(FILTER _runtime_dirs INCLUDE REGEX ".+")
        list(REMOVE_DUPLICATES _runtime_dirs)
    endif()
    set(${out_var} "${_runtime_dirs}" PARENT_SCOPE)
endfunction()

function(rcwa_add_test_with_runtime test_name target_name)
    rcwa_collect_windows_runtime_dirs(_rcwa_runtime_dirs)
    if(WIN32 AND _rcwa_runtime_dirs)
        cmake_path(CONVERT "${_rcwa_runtime_dirs}" TO_NATIVE_PATH_LIST _rcwa_runtime_path NORMALIZE)
        string(REPLACE ";" "\\;" _rcwa_runtime_path "${_rcwa_runtime_path}")
        string(REPLACE ";" "\\;" _rcwa_host_path "$ENV{PATH}")
        add_test(
            NAME ${test_name}
            COMMAND "${CMAKE_COMMAND}" -E env
                "PATH=${_rcwa_runtime_path}\;${_rcwa_host_path}"
                "MKL_NUM_THREADS=1"
                "OMP_NUM_THREADS=1"
                "MKL_THREADING_LAYER=SEQUENTIAL"
                "$<TARGET_FILE:${target_name}>")
    else()
        add_test(NAME ${test_name} COMMAND ${target_name})
    endif()
endfunction()

if(RCWA_SINGLE_PRECISION)
    target_compile_definitions(rcwa PUBLIC RCWA_SINGLE_PRECISION=1)
endif()

if(RCWA_HAS_LAPACKE)
    target_compile_definitions(rcwa PUBLIC RCWA_HAS_LAPACKE=1)
    if(NOT RCWA_SINGLE_PRECISION)
        target_compile_definitions(rcwa PRIVATE LAPACK_COMPLEX_STRUCTURE=1)
    endif()
    target_link_libraries(rcwa PRIVATE ${RCWA_LAPACKE_TARGET})
else()
    message(FATAL_ERROR "RCWA: LAPACKE/BLAS/LAPACK backend is required")
endif()

if(RCWA_HAS_FFTW)
    target_compile_definitions(rcwa PRIVATE RCWA_HAS_FFTW=1)
    if(RCWA_FFTW_INCLUDE_DIR)
        target_include_directories(rcwa PRIVATE ${RCWA_FFTW_INCLUDE_DIR})
    endif()
    target_link_libraries(rcwa PRIVATE ${RCWA_FFTW_TARGET})
else()
    message(FATAL_ERROR "RCWA: FFTW3 backend is required")
endif()

# Tests

enable_testing()

add_executable(rcwaTests tests/rcwaTests.cpp)
target_link_libraries(rcwaTests PRIVATE rcwa Catch2::Catch2WithMain)
rcwa_add_test_with_runtime(rcwaTests rcwaTests)

# Python bindings

if(RCWA_ENABLE_PYTHON_BINDINGS)
    find_package(Python3 COMPONENTS Interpreter Development.Module NumPy REQUIRED)
    execute_process(
        COMMAND "${Python3_EXECUTABLE}" -m pybind11 --cmakedir
        OUTPUT_VARIABLE RCWA_PYBIND11_CMAKE_DIR
        OUTPUT_STRIP_TRAILING_WHITESPACE
        RESULT_VARIABLE RCWA_PYBIND11_CMAKE_DIR_RESULT
    )
    if(RCWA_PYBIND11_CMAKE_DIR_RESULT EQUAL 0 AND EXISTS "${RCWA_PYBIND11_CMAKE_DIR}")
        list(APPEND CMAKE_PREFIX_PATH "${RCWA_PYBIND11_CMAKE_DIR}")
    endif()
    find_package(pybind11 CONFIG REQUIRED)
    pybind11_add_module(rcwa_cpp
        bindings/rcwaCpp.cpp
        bindings/rcwaBindingUtils.cpp
        bindings/rcwaSimulation.cpp
    )
    rcwa_enable_release_ipo(rcwa_cpp)
    target_link_libraries(rcwa_cpp PRIVATE rcwa Python3::NumPy)
    set(_rcwa_python_output_dir "${CMAKE_CURRENT_BINARY_DIR}/python")
    set_target_properties(rcwa_cpp PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY "${_rcwa_python_output_dir}"
        RUNTIME_OUTPUT_DIRECTORY "${_rcwa_python_output_dir}"
        ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
    )
    foreach(_rcwa_config DEBUG RELEASE RELWITHDEBINFO MINSIZEREL)
        set_target_properties(rcwa_cpp PROPERTIES
            LIBRARY_OUTPUT_DIRECTORY_${_rcwa_config} "${_rcwa_python_output_dir}"
            RUNTIME_OUTPUT_DIRECTORY_${_rcwa_config} "${_rcwa_python_output_dir}"
        )
    endforeach()

    # Keep the binding target buildable for downstream users that do not ship
    # the optional repository-level Python smoke test.  Register it only when
    # the script is present; otherwise CTest would report a false backend
    # failure before executing any code.
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/pythonBindingTests.py")
        rcwa_collect_windows_runtime_dirs(_rcwa_python_runtime_dirs)
        if(WIN32 AND _rcwa_python_runtime_dirs)
            cmake_path(
                CONVERT "${_rcwa_python_runtime_dirs}"
                TO_NATIVE_PATH_LIST _rcwa_python_runtime_path
                NORMALIZE)
            string(REPLACE ";" "\;" _rcwa_python_runtime_path "${_rcwa_python_runtime_path}")
            string(REPLACE ";" "\;" _rcwa_python_host_path "$ENV{PATH}")
            add_test(
                NAME rcwaPythonBindingTests
                COMMAND "${CMAKE_COMMAND}" -E env
                    "PATH=${_rcwa_python_runtime_path}\;${_rcwa_python_host_path}"
                    "MKL_NUM_THREADS=1"
                    "OMP_NUM_THREADS=1"
                    "MKL_THREADING_LAYER=SEQUENTIAL"
                    "${Python3_EXECUTABLE}"
                    "${CMAKE_CURRENT_SOURCE_DIR}/tests/pythonBindingTests.py"
                    "$<TARGET_FILE_DIR:rcwa_cpp>"
                    "${CMAKE_CURRENT_SOURCE_DIR}")
        else()
            add_test(
                NAME rcwaPythonBindingTests
                COMMAND "${Python3_EXECUTABLE}"
                    "${CMAKE_CURRENT_SOURCE_DIR}/tests/pythonBindingTests.py"
                    "$<TARGET_FILE_DIR:rcwa_cpp>"
                    "${CMAKE_CURRENT_SOURCE_DIR}")
        endif()
    else()
        message(STATUS "RCWA: optional tests/pythonBindingTests.py not present; skipping binding smoke test")
    endif()
endif()
