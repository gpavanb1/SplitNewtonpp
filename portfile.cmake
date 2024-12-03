vcpkg_configure_cmake(
    SOURCE_PATH ${CURRENT_PORT_DIR}/../SplitNewtonpp
    PREFER_NINJA
)

vcpkg_install_cmake()

# Manually move CMake configuration files to the correct location
file(INSTALL
    DESTINATION ${CURRENT_PACKAGES_DIR}/share/splitnewtonpp/cmake
    TYPE FILE
    FILES ${CURRENT_PACKAGES_DIR}/lib/cmake/SplitNewton/SplitNewtonConfig.cmake
)

# Clean up the original CMake directories
file(REMOVE_RECURSE ${CURRENT_PACKAGES_DIR}/lib/cmake)

# Remove unnecessary debug/include directory
file(REMOVE_RECURSE "${CURRENT_PACKAGES_DIR}/debug/include")

