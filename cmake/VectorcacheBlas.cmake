# CBLAS (SGEMM) for landmark projection. Prefers system OpenBLAS/MKL;
# on Windows fetches the official OpenBLAS x64 binary if none is found.

set(VECTORCACHE_OPENBLAS_DLL "")
set(VECTORCACHE_BLAS_LIBRARIES "")
set(VECTORCACHE_CBLAS_INCLUDE_DIR "")

# HPC modules often expose headers only via CPATH / C_INCLUDE_PATH, which
# find_path does not search by default.
set(_vc_cblas_path_hints "")
foreach(_vc_env OPENBLAS_ROOT OpenBLAS_ROOT OPENBLAS_HOME OPENBLAS_DIR OpenBLAS_DIR MKLROOT)
  if(DEFINED ENV{${_vc_env}} AND NOT "$ENV{${_vc_env}}" STREQUAL "")
    list(APPEND _vc_cblas_path_hints
      "$ENV{${_vc_env}}"
      "$ENV{${_vc_env}}/include")
  endif()
endforeach()
foreach(_vc_env CPATH C_INCLUDE_PATH CPLUS_INCLUDE_PATH INCLUDE)
  if(DEFINED ENV{${_vc_env}} AND NOT "$ENV{${_vc_env}}" STREQUAL "")
    string(REPLACE ":" ";" _vc_env_list "$ENV{${_vc_env}}")
    list(APPEND _vc_cblas_path_hints ${_vc_env_list})
  endif()
endforeach()
foreach(_vc_env LIBRARY_PATH LD_LIBRARY_PATH)
  if(DEFINED ENV{${_vc_env}} AND NOT "$ENV{${_vc_env}}" STREQUAL "")
    string(REPLACE ":" ";" _vc_env_list "$ENV{${_vc_env}}")
    foreach(_vc_libdir ${_vc_env_list})
      get_filename_component(_vc_parent "${_vc_libdir}" DIRECTORY)
      list(APPEND _vc_cblas_path_hints "${_vc_parent}" "${_vc_parent}/include" "${_vc_libdir}/../include")
    endforeach()
  endif()
endforeach()
list(APPEND _vc_cblas_path_hints
  /usr
  /usr/local
  /opt/OpenBLAS
  /usr/include
  /usr/local/include
  /opt/OpenBLAS/include)
list(REMOVE_DUPLICATES _vc_cblas_path_hints)

# Drop stale cache from earlier configures without module env.
unset(VECTORCACHE_CBLAS_INCLUDE_DIR CACHE)
unset(VECTORCACHE_BLAS_LIBRARIES CACHE)

function(_vc_find_cblas_header out_var)
  unset(${out_var} CACHE)
  find_path(${out_var}
    NAMES cblas.h
    HINTS ${_vc_cblas_path_hints}
    PATH_SUFFIXES include openblas include/openblas)
endfunction()

# 1) Explicit OpenBLAS root with both header + library.
foreach(_vc_root "$ENV{OPENBLAS_ROOT}" "$ENV{OpenBLAS_ROOT}" "$ENV{OPENBLAS_HOME}" "$ENV{OPENBLAS_DIR}")
  if(_vc_root AND EXISTS "${_vc_root}/include/cblas.h")
    unset(_vc_ob_lib CACHE)
    find_library(_vc_ob_lib
      NAMES openblas libopenblas
      PATHS "${_vc_root}/lib" "${_vc_root}/lib64"
      NO_DEFAULT_PATH)
    if(_vc_ob_lib)
      set(VECTORCACHE_BLAS_LIBRARIES "${_vc_ob_lib}")
      set(VECTORCACHE_CBLAS_INCLUDE_DIR "${_vc_root}/include")
      message(STATUS "vectorcache: using OpenBLAS at ${_vc_root}")
      break()
    endif()
  endif()
endforeach()

# 2) MKL
if(NOT VECTORCACHE_BLAS_LIBRARIES AND DEFINED ENV{MKLROOT})
  unset(_vc_mkl_cblas CACHE)
  find_path(_vc_mkl_cblas NAMES cblas.h mkl_cblas.h
    PATHS "$ENV{MKLROOT}/include" NO_DEFAULT_PATH)
  if(_vc_mkl_cblas)
    set(BLA_VENDOR "Intel10_64lp" CACHE STRING "" FORCE)
    find_package(BLAS QUIET)
    if(BLAS_FOUND)
      set(VECTORCACHE_BLAS_LIBRARIES ${BLAS_LIBRARIES})
      set(VECTORCACHE_CBLAS_INCLUDE_DIR "${_vc_mkl_cblas}")
      message(STATUS "vectorcache: using MKL BLAS (${BLAS_LIBRARIES})")
    endif()
  endif()
endif()

# 3) FindBLAS + header from hints / CPATH
if(NOT VECTORCACHE_BLAS_LIBRARIES)
  if(NOT BLA_VENDOR)
    set(BLA_VENDOR "OpenBLAS")
  endif()
  find_package(BLAS QUIET)
  _vc_find_cblas_header(_vc_cblas_inc)
  if(BLAS_FOUND AND _vc_cblas_inc)
    set(VECTORCACHE_BLAS_LIBRARIES ${BLAS_LIBRARIES})
    set(VECTORCACHE_CBLAS_INCLUDE_DIR "${_vc_cblas_inc}")
    message(STATUS "vectorcache: using system BLAS (${BLAS_LIBRARIES})")
    message(STATUS "vectorcache: cblas.h in ${VECTORCACHE_CBLAS_INCLUDE_DIR}")
  endif()
endif()

# 4) OpenBLAS CMake package
if(NOT VECTORCACHE_BLAS_LIBRARIES)
  find_package(OpenBLAS CONFIG QUIET)
  if(OpenBLAS_FOUND OR TARGET OpenBLAS::OpenBLAS)
    if(TARGET OpenBLAS::OpenBLAS)
      set(VECTORCACHE_BLAS_LIBRARIES OpenBLAS::OpenBLAS)
      get_target_property(_vc_ob_inc OpenBLAS::OpenBLAS INTERFACE_INCLUDE_DIRECTORIES)
      if(_vc_ob_inc AND NOT _vc_ob_inc STREQUAL "_vc_ob_inc-NOTFOUND")
        list(GET _vc_ob_inc 0 VECTORCACHE_CBLAS_INCLUDE_DIR)
      endif()
    elseif(DEFINED OpenBLAS_LIBRARIES)
      set(VECTORCACHE_BLAS_LIBRARIES ${OpenBLAS_LIBRARIES})
    endif()
    if(NOT VECTORCACHE_CBLAS_INCLUDE_DIR AND DEFINED OpenBLAS_INCLUDE_DIRS)
      list(GET OpenBLAS_INCLUDE_DIRS 0 VECTORCACHE_CBLAS_INCLUDE_DIR)
    endif()
    if(VECTORCACHE_BLAS_LIBRARIES AND NOT VECTORCACHE_CBLAS_INCLUDE_DIR)
      _vc_find_cblas_header(VECTORCACHE_CBLAS_INCLUDE_DIR)
    endif()
    if(VECTORCACHE_BLAS_LIBRARIES AND VECTORCACHE_CBLAS_INCLUDE_DIR)
      message(STATUS "vectorcache: using OpenBLAS CMake package")
    else()
      set(VECTORCACHE_BLAS_LIBRARIES "")
      set(VECTORCACHE_CBLAS_INCLUDE_DIR "")
    endif()
  endif()
endif()

# 5) pkg-config
if(NOT VECTORCACHE_BLAS_LIBRARIES)
  find_package(PkgConfig QUIET)
  if(PkgConfig_FOUND)
    pkg_check_modules(PC_OPENBLAS QUIET openblas)
    if(PC_OPENBLAS_FOUND)
      set(VECTORCACHE_BLAS_LIBRARIES ${PC_OPENBLAS_LINK_LIBRARIES})
      if(PC_OPENBLAS_INCLUDE_DIRS)
        list(GET PC_OPENBLAS_INCLUDE_DIRS 0 VECTORCACHE_CBLAS_INCLUDE_DIR)
      endif()
      if(NOT VECTORCACHE_CBLAS_INCLUDE_DIR)
        _vc_find_cblas_header(VECTORCACHE_CBLAS_INCLUDE_DIR)
      endif()
      if(VECTORCACHE_BLAS_LIBRARIES AND VECTORCACHE_CBLAS_INCLUDE_DIR)
        message(STATUS "vectorcache: using pkg-config OpenBLAS")
      else()
        set(VECTORCACHE_BLAS_LIBRARIES "")
        set(VECTORCACHE_CBLAS_INCLUDE_DIR "")
      endif()
    endif()
  endif()
endif()

# 6) Header found via CPATH but FindBLAS missed the lib — locate libopenblas near it.
if(NOT VECTORCACHE_BLAS_LIBRARIES)
  _vc_find_cblas_header(_vc_cblas_inc)
  if(_vc_cblas_inc)
    get_filename_component(_vc_inc_parent "${_vc_cblas_inc}" DIRECTORY)
    unset(_vc_ob_lib CACHE)
    find_library(_vc_ob_lib
      NAMES openblas libopenblas
      HINTS "${_vc_inc_parent}/lib" "${_vc_inc_parent}/lib64" ${_vc_cblas_path_hints}
      PATH_SUFFIXES lib lib64)
    if(_vc_ob_lib)
      set(VECTORCACHE_BLAS_LIBRARIES "${_vc_ob_lib}")
      set(VECTORCACHE_CBLAS_INCLUDE_DIR "${_vc_cblas_inc}")
      message(STATUS "vectorcache: using OpenBLAS lib ${_vc_ob_lib}")
      message(STATUS "vectorcache: cblas.h in ${VECTORCACHE_CBLAS_INCLUDE_DIR}")
    endif()
  endif()
endif()

if(VECTORCACHE_BLAS_LIBRARIES AND VECTORCACHE_CBLAS_INCLUDE_DIR)
  if(NOT EXISTS "${VECTORCACHE_CBLAS_INCLUDE_DIR}/cblas.h"
     AND NOT EXISTS "${VECTORCACHE_CBLAS_INCLUDE_DIR}/mkl_cblas.h")
    message(FATAL_ERROR
      "vectorcache: CBLAS include dir '${VECTORCACHE_CBLAS_INCLUDE_DIR}' has no cblas.h")
  endif()
elseif(WIN32)
  set(_vc_ob_ver "0.3.34")
  set(_vc_ob_root "${CMAKE_BINARY_DIR}/_deps/openblas-x64")
  set(_vc_ob_zip "${CMAKE_BINARY_DIR}/_deps/OpenBLAS-${_vc_ob_ver}-x64.zip")
  set(_vc_ob_url
    "https://github.com/OpenMathLib/OpenBLAS/releases/download/v${_vc_ob_ver}/OpenBLAS-${_vc_ob_ver}-x64.zip")

  if(NOT EXISTS "${_vc_ob_root}/include/cblas.h")
    message(STATUS "vectorcache: fetching OpenBLAS ${_vc_ob_ver} (Windows x64)")
    file(DOWNLOAD "${_vc_ob_url}" "${_vc_ob_zip}" SHOW_PROGRESS
         STATUS _vc_dl_status TLS_VERIFY ON)
    list(GET _vc_dl_status 0 _vc_dl_code)
    if(NOT _vc_dl_code EQUAL 0)
      message(FATAL_ERROR "vectorcache: OpenBLAS download failed (${_vc_dl_status})")
    endif()
    file(ARCHIVE_EXTRACT INPUT "${_vc_ob_zip}" DESTINATION "${_vc_ob_root}")
  endif()

  if(NOT EXISTS "${_vc_ob_root}/include/cblas.h" OR NOT EXISTS "${_vc_ob_root}/lib/libopenblas.lib")
    message(FATAL_ERROR "vectorcache: OpenBLAS extract incomplete at ${_vc_ob_root}")
  endif()
  set(VECTORCACHE_CBLAS_INCLUDE_DIR "${_vc_ob_root}/include")
  set(VECTORCACHE_BLAS_LIBRARIES "${_vc_ob_root}/lib/libopenblas.lib")
  set(VECTORCACHE_OPENBLAS_DLL "${_vc_ob_root}/bin/libopenblas.dll")
  message(STATUS "vectorcache: using bundled OpenBLAS at ${_vc_ob_root}")
else()
  message(FATAL_ERROR
    "vectorcache: CBLAS required for landmark projection.\n"
    "  Install OpenBLAS (e.g. apt install libopenblas-dev) or MKL,\n"
    "  or: module load openblas && export OPENBLAS_ROOT=...\n"
    "  CPATH/LIBRARY_PATH are searched; scripts/envs.sh sets OPENBLAS_ROOT when possible.\n"
    "  Override module name with VECTORCACHE_MODULE_OPENBLAS.")
endif()

function(vectorcache_copy_openblas_dll target_name)
  if(VECTORCACHE_OPENBLAS_DLL AND EXISTS "${VECTORCACHE_OPENBLAS_DLL}")
    add_custom_command(TARGET ${target_name} POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E copy_if_different
              "${VECTORCACHE_OPENBLAS_DLL}"
              "$<TARGET_FILE_DIR:${target_name}>"
      COMMENT "Copy libopenblas.dll for ${target_name}")
  endif()
endfunction()
