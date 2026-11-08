cmake_minimum_required(VERSION 3.18)
project(NeuralLanguageMobile LANGUAGES CXX)
add_library(neuralmobile SHARED ../mobile_infer.cpp neural_jni.cpp)
target_compile_features(neuralmobile PRIVATE cxx_std_17)
target_compile_definitions(neuralmobile PRIVATE NL_MOBILE_LIBRARY_ONLY)
target_compile_options(neuralmobile PRIVATE -O2 -fexceptions -Wall -Wextra -Werror)
# Android NDK CMake provides JNI headers automatically. No third-party library.
# On desktop add JDK jni.h include locations to the target when invoking CMake.
