# CBLAS (SGEMM) for landmark projection. Prefers system OpenBLAS/MKL;
# on Windows fetches the official OpenBLAS x64 binary if none is found.

set(VECTORCACHE_OPENBLAS_DLL "")
set(VECTORCACHE_BLAS_LIBRARIES "")
set(VECTORCACHE_CBLAS_INCLUDE_DIR "")

# Honor explicit roots before FindBLAS (HPC modules often set these).
foreach(_vc_root "$ENV{OPENBLAS_ROOT}" "$ENV{OpenBLAS_ROOT}" "$ENV{OPENBLAS_HOME}")
  if(_vc_root AND EXISTS "${_vc_root}/include/cblas.h")
    find_library(VECTORCACHE_BLAS_LIBRARIES
      NAMES openblas libopenblas
      PATHS "${_vc_root}/lib" "${_vc_root}/lib64"
      NO_DEFAULT_PATH)
    if(VECTORCACHE_BLAS_LIBRARIES)
      set(VECTORCACHE_CBLAS_INCLUDE_DIR "${_vc_root}/include")
      message(STATUS "vectorcache: using OpenBLAS at ${_vc_root}")
      break()
    endif()
  endif()
endforeach()

if(NOT VECTORCACHE_BLAS_LIBRARIES AND DEFINED ENV{MKLROOT})
  find_path(_vc_mkl_cblas NAMES cblas.h mkl_cblas.h PATHS "$ENV{MKLROOT}/include" NO_DEFAULT_PATH)
  if(_vc_mkl_cblas)
    find_package(BLAS QUIET)
    if(BLAS_FOUND)
      set(VECTORCACHE_BLAS_LIBRARIES ${BLAS_LIBRARIES})
      set(VECTORCACHE_CBLAS_INCLUDE_DIR "${_vc_mkl_cblas}")
      message(STATUS "vectorcache: using MKL BLAS (${BLAS_LIBRARIES})")
    endif()
  endif()
endif()

if(NOT VECTORCACHE_BLAS_LIBRARIES)
  # Prefer OpenBLAS when FindBLAS has several vendors available.
  if(NOT BLA_VENDOR)
    set(BLA_VENDOR "OpenBLAS")
  endif()
  find_package(BLAS QUIET)
  find_path(VECTORCACHE_CBLAS_INCLUDE_DIR
    NAMES cblas.h
    PATHS
      ${BLAS_INCLUDE_DIRS}
      $ENV{OPENBLAS_ROOT}/include
      $ENV{OpenBLAS_ROOT}/include
      $ENV{OPENBLAS_HOME}/include
      $ENV{MKLROOT}/include
      /usr/include
      /usr/local/include
      /opt/OpenBLAS/include
    PATH_SUFFIXES openblas
  )
  if(BLAS_FOUND AND VECTORCACHE_CBLAS_INCLUDE_DIR)
    set(VECTORCACHE_BLAS_LIBRARIES ${BLAS_LIBRARIES})
    message(STATUS "vectorcache: using system BLAS (${BLAS_LIBRARIES})")
  endif()
endif()

if(NOT VECTORCACHE_BLAS_LIBRARIES)
  find_package(OpenBLAS CONFIG QUIET)
  if(OpenBLAS_FOUND)
    if(TARGET OpenBLAS::OpenBLAS)
      set(VECTORCACHE_BLAS_LIBRARIES OpenBLAS::OpenBLAS)
    elseif(DEFINED OpenBLAS_LIBRARIES)
      set(VECTORCACHE_BLAS_LIBRARIES ${OpenBLAS_LIBRARIES})
    endif()
    if(DEFINED OpenBLAS_INCLUDE_DIRS)
      list(GET OpenBLAS_INCLUDE_DIRS 0 VECTORCACHE_CBLAS_INCLUDE_DIR)
    endif()
    if(VECTORCACHE_BLAS_LIBRARIES AND NOT VECTORCACHE_CBLAS_INCLUDE_DIR)
      find_path(VECTORCACHE_CBLAS_INCLUDE_DIR NAMES cblas.h
        PATHS ${OpenBLAS_INCLUDE_DIRS} PATH_SUFFIXES openblas)
    endif()
    if(VECTORCACHE_BLAS_LIBRARIES AND VECTORCACHE_CBLAS_INCLUDE_DIR)
      message(STATUS "vectorcache: using OpenBLAS CMake package")
    else()
      set(VECTORCACHE_BLAS_LIBRARIES "")
      set(VECTORCACHE_CBLAS_INCLUDE_DIR "")
    endif()
  endif()
endif()

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
        find_path(VECTORCACHE_CBLAS_INCLUDE_DIR NAMES cblas.h
          HINTS ${PC_OPENBLAS_INCLUDEDIR} ${PC_OPENBLAS_INCLUDE_DIRS}
          PATH_SUFFIXES openblas)
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

if(VECTORCACHE_BLAS_LIBRARIES AND VECTORCACHE_CBLAS_INCLUDE_DIR)
  # already configured
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
    "  or set OPENBLAS_ROOT / MKLROOT, or: module load openblas\n"
    "  (scripts/envs.sh tries VECTORCACHE_MODULE_OPENBLAS).")
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
