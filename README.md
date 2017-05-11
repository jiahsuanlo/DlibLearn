# DlibLearn
dlib practice

# Use CUDA Support in dlib

- Create Visual C++ project using cmake
- Build dlib project to create dlib.lib
- Create application project
- include path: add dlib main path
- library path: add dlib lib, CUDA, and cudnn paths
- linker input: dlib.lib; cudnn.lib; curand.lib; cudart.lib; cublas.lib;
- preprocessor: add DLIB_USE_CUDA 
