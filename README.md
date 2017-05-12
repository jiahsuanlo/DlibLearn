# DlibLearn
dlib practice

# Use CUDA Support in dlib

1. Create Visual C++ project using cmake
2. Build dlib project to create dlib.lib
3. Create application project
4. include path: add dlib main path
5. library path: add dlib lib, CUDA, and cudnn paths
6. linker input: dlib.lib; cudnn.lib; curand.lib; cudart.lib; cublas.lib;
7. preprocessor: add DLIB_USE_CUDA 

Steps 4-7 can be set up by using dlibPropertySheet.props