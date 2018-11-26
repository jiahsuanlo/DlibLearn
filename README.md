# DlibLearn
dlib practice codes. 

# Installation:
The application projects can be generated using cmake. Please refer to the next section for incorporating CUDA support in dlib.

# Use CUDA Support in dlib

1. Create Visual C++ project using cmake gui
2. Build dlib project to create dlib.lib
3. Create application project
4. include path: add dlib main path
5. library path: add dlib lib, CUDA, and cudnn paths
6. linker input: dlib.lib; cudnn.lib; curand.lib; cudart.lib; cublas.lib;
7. preprocessor: add DLIB_USE_CUDA 
