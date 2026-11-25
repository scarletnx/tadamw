# SPDX-License-Identifier: GPL-3.0-or-later
# libtadamw. One set of objects builds both the public library (static or
# shared, only the TAW_API functions exported) and tadamw_internal, a static
# copy the tests link to reach the internals.
if(TADAMW_ENCODER)
  set(TAW_HAVE_ENCODER 1)
else()
  set(TAW_HAVE_ENCODER 0)
endif()
if(TADAMW_DECODER)
  set(TAW_HAVE_DECODER 1)
else()
  set(TAW_HAVE_DECODER 0)
endif()
if(TADAMW_ENCODER AND TADAMW_GPU)
  if(NOT TADAMW_VULKAN_INCLUDE_DIR OR NOT TADAMW_GLSLANG)
    message(FATAL_ERROR "TADAMW_GPU needs vulkan/vulkan.h and glslangValidator")
  endif()
  set(TAW_HAVE_GPU 1)
else()
  set(TAW_HAVE_GPU 0)
endif()
configure_file(${PROJECT_SOURCE_DIR}/cmake/config.h.in
               ${PROJECT_BINARY_DIR}/include/tadamw/config.h @ONLY)

set(tadamw_sources
  api.c aw.c awseek.c buf.c crc32c.c header.c p1_ctx.c p1_net.c pool.c
  predictor.c quant.c rc.c version.c)
if(TADAMW_ENCODER)
  list(APPEND tadamw_sources fit.c fit_opt.c fit_gpu.c) # AdamW fitting, encoder only
endif()
if(TAW_HAVE_GPU)
  # The trainer's compute shader, as SPIR-V words in a header.
  set(gen ${PROJECT_BINARY_DIR}/gen)
  foreach(shader fit_gpu fit_gpu_prep)
    add_custom_command(OUTPUT ${gen}/${shader}_spv.h
      COMMAND ${CMAKE_COMMAND} -E make_directory ${gen}
      COMMAND ${TADAMW_GLSLANG} -V --target-env vulkan1.1 --vn ${shader}_spv
              -o ${gen}/${shader}_spv.h ${CMAKE_CURRENT_SOURCE_DIR}/${shader}.comp
      DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/${shader}.comp
      COMMENT "glslangValidator ${shader}.comp" VERBATIM)
    list(APPEND tadamw_sources ${gen}/${shader}_spv.h)
  endforeach()
endif()

add_library(tadamw_obj OBJECT ${tadamw_sources})
set_target_properties(tadamw_obj PROPERTIES POSITION_INDEPENDENT_CODE ON
                      C_VISIBILITY_PRESET hidden)
target_include_directories(tadamw_obj
  PUBLIC  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
          $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/include>
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_definitions(tadamw_obj PRIVATE TAW_GPU=${TAW_HAVE_GPU})
if(TAW_HAVE_GPU)
  target_include_directories(tadamw_obj PRIVATE ${PROJECT_BINARY_DIR}/gen
                             ${TADAMW_VULKAN_INCLUDE_DIR})
endif()

# x86 SIMD kernels are compiled with their own ISA flags and picked at run
# time by CPUID; everything else stays baseline.
if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|amd64|AMD64)$")
  target_sources(tadamw_obj PRIVATE p1_avx2.c p1_run_avx2.c p1_avx512.c crc32c_sse42.c)
  set_source_files_properties(p1_avx2.c p1_run_avx2.c PROPERTIES COMPILE_OPTIONS "-mavx2;-mf16c")
  set_source_files_properties(p1_avx512.c PROPERTIES COMPILE_OPTIONS "-mavx512f;-mavx2;-mf16c")
  set_source_files_properties(crc32c_sse42.c PROPERTIES COMPILE_OPTIONS "-msse4.2")
  target_compile_definitions(tadamw_obj PRIVATE TAW_X86_SIMD=1)
else()
  target_compile_definitions(tadamw_obj PRIVATE TAW_X86_SIMD=0)
endif()

find_package(Threads REQUIRED) # .awseek fitting and block coding (pool.c)

add_library(tadamw $<TARGET_OBJECTS:tadamw_obj>)
add_library(tadamw::tadamw ALIAS tadamw)
set_target_properties(tadamw PROPERTIES
  VERSION ${PROJECT_VERSION} SOVERSION ${PROJECT_VERSION_MAJOR}
  C_VISIBILITY_PRESET hidden)
target_include_directories(tadamw PUBLIC
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
  $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/include>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)
target_link_libraries(tadamw PRIVATE Threads::Threads ${CMAKE_DL_LIBS})

add_library(tadamw_internal STATIC $<TARGET_OBJECTS:tadamw_obj>)
target_include_directories(tadamw_internal PUBLIC
  ${PROJECT_SOURCE_DIR}/include ${PROJECT_BINARY_DIR}/include
  ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(tadamw_internal PUBLIC Threads::Threads ${CMAKE_DL_LIBS})

if(TADAMW_INSTALL)
  include(CMakePackageConfigHelpers)
  install(TARGETS tadamw EXPORT tadamwTargets
          ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
          LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
          RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
  install(FILES ${PROJECT_SOURCE_DIR}/include/tadamw/tadamw.h
                ${PROJECT_BINARY_DIR}/include/tadamw/config.h
          DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/tadamw)
  set(cmake_dir ${CMAKE_INSTALL_LIBDIR}/cmake/tadamw)
  install(EXPORT tadamwTargets NAMESPACE tadamw:: DESTINATION ${cmake_dir})
  configure_package_config_file(${PROJECT_SOURCE_DIR}/cmake/tadamwConfig.cmake.in
    ${PROJECT_BINARY_DIR}/tadamwConfig.cmake INSTALL_DESTINATION ${cmake_dir})
  write_basic_package_version_file(${PROJECT_BINARY_DIR}/tadamwConfigVersion.cmake
    COMPATIBILITY SameMinorVersion)
  install(FILES ${PROJECT_BINARY_DIR}/tadamwConfig.cmake
                ${PROJECT_BINARY_DIR}/tadamwConfigVersion.cmake
          DESTINATION ${cmake_dir})
  configure_file(${PROJECT_SOURCE_DIR}/cmake/tadamw.pc.in
                 ${PROJECT_BINARY_DIR}/tadamw.pc @ONLY)
  install(FILES ${PROJECT_BINARY_DIR}/tadamw.pc
          DESTINATION ${CMAKE_INSTALL_LIBDIR}/pkgconfig)
endif()
