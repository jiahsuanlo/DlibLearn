cmake_minimum_required(VERSION 2.8.12)
# Every project needs a testDeep.  We call this the "examples" project.
project(DlibLearn)


# Tell cmake we will need dlib.  This command will pull in dlib and compile it
# into your project.  Note that you don't need to compile or install dlib.  All
# cmake needs is the dlib source code folder and it will take care of everything.
include(C:/dev/dlib-19.7/dlib/cmake)


# The next thing we need to do is tell CMake about the code you want to
# compile.  We do this with the add_executable() statement which takes the testDeep
# of the output executable and then a list of .cpp files to compile.  Here we
# are going to compile one of the dlib example programs which has only one .cpp
# file, assignment_learning_ex.cpp.  If your program consisted of multiple .cpp
# files you would simply list them here in the add_executable() statement.  
add_executable(DeepBasic ./src/DeepBasic/deepBasic.cpp)
# Finally, you need to tell CMake that this program, assignment_learning_ex,
# depends on dlib.  You do that with this statement: 
target_link_libraries(DeepBasic dlib::dlib)

add_executable(PtCloud ./src/PtCloud/ptCloud.cpp)
target_link_libraries(PtCloud dlib::dlib)

add_executable(DeepAdvanced ./src/DeepAdvanced/deepAdvanced.cpp)
target_link_libraries(DeepAdvanced dlib::dlib)

add_executable(VehicleDetector ./src/VehicleDetector/vehicleDetection.cpp)
target_link_libraries(VehicleDetector dlib::dlib)

add_executable(MaxMarginObjectDetection ./src/MaxMarginObjectDetection/maxMargin.cpp)
target_link_libraries(MaxMarginObjectDetection dlib::dlib)

add_executable(VehicleFrontBackDetector ./src/VehicleFrontBackDetector/vehicleFrontBackDetector.cpp)
target_link_libraries(VehicleFrontBackDetector dlib::dlib)