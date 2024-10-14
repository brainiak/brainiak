add_custom_command(
  OUTPUT _utils.c
  COMMENT
    "Making ${CMAKE_CURRENT_BINARY_DIR}/_utils.c from ${CMAKE_CURRENT_SOURCE_DIR}/_utils.pyx"
  COMMAND Python::Interpreter -m cython
          "${CMAKE_CURRENT_SOURCE_DIR}/_utils.pyx" --output-file _utils.c
  DEPENDS _utils.pyx
  VERBATIM)

python_add_library(_utils MODULE _utils.c WITH_SOABI)
target_link_libraries(_utils PUBLIC Python::NumPy)

install(TARGETS _utils DESTINATION ${SKBUILD_PROJECT_NAME}/eventseg)
