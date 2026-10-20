# CBLAS (SGEMM) for landmark projection. Prefers a system BLAS; on Windows
# fetches the official OpenBLAS x64 binary if none is found.

set(VECTORCACHE_OPENBLAS_DLL "")
set(VECTORCACHE_BLAS_LIBRARIES "")
set(VECTORCACHE_CBLAS_INCLUDE_DIR "")

find_package(BLAS QUIET)
find_path(VECTORCACHE_CBLAS_INCLUDE_DIR
  NAMES cblas.h
  PATHS
    ${BLAS_INCLUDE_DIRS}
    $ENV{OPENBLAS_ROOT}/include
    $ENV{OpenBLAS_ROOT}/include
    $ENV{MKLROOT}/include
  PATH_SUFFIXES openblas
)

if(BLAS_FOUND AND VECTORCACHE_CBLAS_INCLUDE_DIR)
  set(VECTORCACHE_BLAS_LIBRARIES ${BLAS_LIBRARIES})
  message(STATUS "vectorcache: using system BLAS (${BLAS_LIBRARIES})")
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
    "vectorcache: CBLAS required for landmark projection. "
    "Install OpenBLAS/MKL or set OPENBLAS_ROOT / BLA_VENDOR.")
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
