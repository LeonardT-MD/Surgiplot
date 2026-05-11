cmake_minimum_required(VERSION 3.16.3...3.19.7 FATAL_ERROR)

project(SurgiplotToolkit)

find_package(Slicer REQUIRED)
include(${Slicer_USE_FILE})

set(EXTENSION_HOMEPAGE "https://github.com/LeonardT-MD/Surgiplot")
set(EXTENSION_CATEGORY "Neurosurgery")
set(EXTENSION_CONTRIBUTORS "Leonardo Tariciotti")
set(EXTENSION_DESCRIPTION "3D Slicer extension for Surgiplot coordinate-based operative corridor analysis.")
set(EXTENSION_ICONURL "")
set(EXTENSION_SCREENSHOTURLS "")
set(EXTENSION_DEPENDS "Markups")

add_subdirectory(SurgiplotToolkit)

include(${Slicer_EXTENSION_GENERATE_CONFIG})
include(${Slicer_EXTENSION_CPACK})
