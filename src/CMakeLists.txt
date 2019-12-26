#根目录CMAKE
cmake_minimum_required(VERSION 3.13)
project(DarkSense)

#set path
set(OpenCV_DIR "D:/Libraries/OpenCV/4.1.2/build_vc16_withcuda/install")
set(Qt5_DIR "D:/Libraries/Qt/5.13.2/msvc2017_64/lib/cmake/Qt5")


add_subdirectory(DarkNet)
add_subdirectory(RealSenseViewer)
