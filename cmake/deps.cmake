include(FetchContent)

function(rcwa_add_mkl)
    set(RCWA_HAS_LAPACKE OFF PARENT_SCOPE)
    set(RCWA_LAPACKE_TARGET "" PARENT_SCOPE)

    find_path(MKL_INCLUDE_DIR
        NAMES mkl_lapacke.h mkl_cblas.h
        HINTS "$ENV{MKLROOT}" "C:/Program Files (x86)/Intel/oneAPI/mkl/latest" "C:/Program Files/Intel/oneAPI/mkl/latest"
        PATH_SUFFIXES include
    )
    find_library(MKL_RT_LIBRARY
        NAMES mkl_rt
        HINTS "$ENV{MKLROOT}" "C:/Program Files (x86)/Intel/oneAPI/mkl/latest" "C:/Program Files/Intel/oneAPI/mkl/latest"
        PATH_SUFFIXES lib lib/intel64
    )
    if(NOT MKL_INCLUDE_DIR OR NOT MKL_RT_LIBRARY)
        message(FATAL_ERROR
            "RCWA: Intel oneAPI MKL is the only supported dense numerical backend. "
            "Set MKLROOT or install oneAPI MKL so mkl_lapacke.h, mkl_cblas.h, and mkl_rt are available.")
    endif()

    add_library(RCWA::LAPACKE INTERFACE IMPORTED GLOBAL)
    target_compile_definitions(RCWA::LAPACKE INTERFACE RCWA_HAS_MKL=1 RCWA_HAS_CBLAS=1)
    target_include_directories(RCWA::LAPACKE INTERFACE ${MKL_INCLUDE_DIR})
    target_link_libraries(RCWA::LAPACKE INTERFACE ${MKL_RT_LIBRARY})
    set(RCWA_HAS_LAPACKE ON PARENT_SCOPE)
    set(RCWA_LAPACKE_TARGET RCWA::LAPACKE PARENT_SCOPE)
    message(STATUS "RCWA: using Intel oneAPI MKL LAPACKE/CBLAS")
endfunction()

function(rcwa_add_fftw)
    set(RCWA_HAS_FFTW OFF PARENT_SCOPE)
    set(RCWA_FFTW_TARGET "" PARENT_SCOPE)
    set(RCWA_FFTW_INCLUDE_DIR "" PARENT_SCOPE)

    find_package(PkgConfig QUIET)
    if(PkgConfig_FOUND)
        if(RCWA_SINGLE_PRECISION)
            pkg_check_modules(FFTW3 QUIET IMPORTED_TARGET fftw3f)
        else()
            pkg_check_modules(FFTW3 QUIET IMPORTED_TARGET fftw3)
        endif()
    endif()
    if(TARGET PkgConfig::FFTW3)
        set(RCWA_HAS_FFTW ON PARENT_SCOPE)
        set(RCWA_FFTW_TARGET PkgConfig::FFTW3 PARENT_SCOPE)
        message(STATUS "RCWA: using system FFTW3 from pkg-config")
        return()
    endif()

    # CMake find_* results are cached. A moved/uninstalled package must not
    # remain truthy and produce an imported target with nonexistent paths.
    foreach(_rcwa_fftw_cached_path FFTW3_INCLUDE_DIR FFTW3_LIBRARY)
        if(DEFINED ${_rcwa_fftw_cached_path} AND
           NOT "${${_rcwa_fftw_cached_path}}" STREQUAL "" AND
           NOT "${${_rcwa_fftw_cached_path}}" MATCHES "-NOTFOUND$" AND
           NOT EXISTS "${${_rcwa_fftw_cached_path}}")
            message(STATUS
                "RCWA: discarding stale ${_rcwa_fftw_cached_path}="
                "${${_rcwa_fftw_cached_path}}")
            unset(${_rcwa_fftw_cached_path} CACHE)
            unset(${_rcwa_fftw_cached_path})
        endif()
    endforeach()

    find_path(FFTW3_INCLUDE_DIR
        NAMES fftw3.h
        HINTS /opt/local /opt/homebrew /usr/local C:/mingw64
        PATH_SUFFIXES include
    )
    if(RCWA_SINGLE_PRECISION)
        set(_rcwa_fftw_names fftw3f libfftw3f)
        set(_rcwa_fftw_target FFTW3::fftw3f)
    else()
        set(_rcwa_fftw_names fftw3 libfftw3)
        set(_rcwa_fftw_target FFTW3::fftw3)
    endif()
    find_library(FFTW3_LIBRARY
        NAMES ${_rcwa_fftw_names}
        HINTS /opt/local /opt/homebrew /usr/local C:/mingw64
        PATH_SUFFIXES lib
    )
    if(FFTW3_INCLUDE_DIR AND FFTW3_LIBRARY)
        add_library(${_rcwa_fftw_target} UNKNOWN IMPORTED GLOBAL)
        set_target_properties(${_rcwa_fftw_target} PROPERTIES
            IMPORTED_LOCATION "${FFTW3_LIBRARY}"
            INTERFACE_INCLUDE_DIRECTORIES "${FFTW3_INCLUDE_DIR}"
        )
        set(RCWA_HAS_FFTW ON PARENT_SCOPE)
        set(RCWA_FFTW_TARGET ${_rcwa_fftw_target} PARENT_SCOPE)
        message(STATUS "RCWA: using system FFTW3")
        return()
    endif()

    # Do not leave half of an old system installation cached while using the
    # fetched target. On a later configure that could be paired accidentally
    # with an unrelated header or library found elsewhere.
    if(FFTW3_INCLUDE_DIR OR FFTW3_LIBRARY)
        message(STATUS "RCWA: discarding incomplete cached system FFTW3 pair")
    endif()
    unset(FFTW3_INCLUDE_DIR CACHE)
    unset(FFTW3_INCLUDE_DIR)
    unset(FFTW3_LIBRARY CACHE)
    unset(FFTW3_LIBRARY)

    message(STATUS "RCWA: FFTW3 not found locally; downloading FFTW 3.3.11")
    if(NOT DEFINED FETCHCONTENT_SOURCE_DIR_FFTW3)
        set(_rcwa_existing_fftw_candidates
            "${CMAKE_BINARY_DIR}/_deps/fftw3-src"
        )
        foreach(_rcwa_existing_fftw IN LISTS _rcwa_existing_fftw_candidates)
            if(EXISTS "${_rcwa_existing_fftw}/api/fftw3.h")
                set(FETCHCONTENT_SOURCE_DIR_FFTW3 "${_rcwa_existing_fftw}" CACHE PATH "" FORCE)
                message(STATUS "RCWA: reusing previously downloaded FFTW3 source")
                break()
            endif()
        endforeach()
    endif()
    set(CMAKE_POLICY_VERSION_MINIMUM 3.10 CACHE STRING "" FORCE)
    set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
    if(RCWA_SINGLE_PRECISION)
        set(ENABLE_FLOAT ON CACHE BOOL "" FORCE)
    else()
        set(ENABLE_FLOAT OFF CACHE BOOL "" FORCE)
    endif()
    set(ENABLE_LONG_DOUBLE OFF CACHE BOOL "" FORCE)
    set(ENABLE_QUAD_PRECISION OFF CACHE BOOL "" FORCE)
    set(ENABLE_THREADS OFF CACHE BOOL "" FORCE)
    set(ENABLE_OPENMP OFF CACHE BOOL "" FORCE)
    set(DISABLE_FORTRAN ON CACHE BOOL "" FORCE)
    set(BUILD_TESTS OFF CACHE BOOL "" FORCE)
    FetchContent_Declare(
        fftw3
        URL https://www.fftw.org/fftw-3.3.11.tar.gz
        URL_HASH SHA256=5630c24cdeb33b131612f7eb4b1a9934234754f9f388ff8617458d0be6f239a1
        DOWNLOAD_EXTRACT_TIMESTAMP TRUE
    )
    FetchContent_MakeAvailable(fftw3)
    if(MSVC)
        foreach(_rcwa_fftw_built_target fftw3 fftw3f)
            if(TARGET ${_rcwa_fftw_built_target})
                foreach(_rcwa_fftw_link_prop LINK_LIBRARIES INTERFACE_LINK_LIBRARIES)
                    get_target_property(_rcwa_fftw_link_libs ${_rcwa_fftw_built_target} ${_rcwa_fftw_link_prop})
                    if(_rcwa_fftw_link_libs)
                        list(REMOVE_ITEM _rcwa_fftw_link_libs m)
                        set_target_properties(${_rcwa_fftw_built_target} PROPERTIES
                            ${_rcwa_fftw_link_prop} "${_rcwa_fftw_link_libs}"
                        )
                    endif()
                endforeach()
            endif()
        endforeach()
    endif()
    if(RCWA_SINGLE_PRECISION AND TARGET fftw3f)
        set(RCWA_HAS_FFTW ON PARENT_SCOPE)
        set(RCWA_FFTW_TARGET fftw3f PARENT_SCOPE)
        set(RCWA_FFTW_INCLUDE_DIR ${fftw3_SOURCE_DIR}/api PARENT_SCOPE)
    elseif((NOT RCWA_SINGLE_PRECISION) AND TARGET fftw3)
        set(RCWA_HAS_FFTW ON PARENT_SCOPE)
        set(RCWA_FFTW_TARGET fftw3 PARENT_SCOPE)
        set(RCWA_FFTW_INCLUDE_DIR ${fftw3_SOURCE_DIR}/api PARENT_SCOPE)
    elseif(RCWA_REQUIRE_FFTW)
        message(FATAL_ERROR "RCWA: FFTW3 is required but could not be found or downloaded")
    endif()
endfunction()

function(rcwa_add_catch2)
    find_package(Catch2 3 QUIET)
    if(NOT Catch2_FOUND)
        message(STATUS "RCWA: Catch2 not found locally; downloading Catch2 3.7.1")
        if(NOT DEFINED FETCHCONTENT_SOURCE_DIR_CATCH2)
            set(_rcwa_existing_catch2_candidates
                "${CMAKE_BINARY_DIR}/_deps/catch2-src"
            )
            foreach(_rcwa_existing_catch2 IN LISTS _rcwa_existing_catch2_candidates)
                if(EXISTS "${_rcwa_existing_catch2}/CMakeLists.txt")
                    set(FETCHCONTENT_SOURCE_DIR_CATCH2 "${_rcwa_existing_catch2}" CACHE PATH "" FORCE)
                    message(STATUS "RCWA: reusing previously downloaded Catch2 source")
                    break()
                endif()
            endforeach()
        endif()
        FetchContent_Declare(
            Catch2
            URL https://github.com/catchorg/Catch2/archive/refs/tags/v3.7.1.tar.gz
            URL_HASH SHA256=c991b247a1a0d7bb9c39aa35faf0fe9e19764213f28ffba3109388e62ee0269c
            DOWNLOAD_EXTRACT_TIMESTAMP TRUE
        )
        FetchContent_MakeAvailable(Catch2)
    endif()
endfunction()
