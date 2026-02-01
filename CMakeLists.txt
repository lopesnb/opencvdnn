cmake_minimum_required(VERSION 3.10.0)
project(opencv_dnn VERSION 0.1.0 LANGUAGES C CXX)


# C++標準を指定 (Boostの利用にはC++11以降を推奨)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# ----------------------------------------------------
# Boostライブラリの検索 (追加)
# ----------------------------------------------------
# system と thread コンポーネントが必要です
find_package(Boost REQUIRED COMPONENTS system thread)

# OpenCVのパッケージを見つける
find_package(OpenCV REQUIRED)

# #OpenCV関係のインクルードディレクトリのパスを設定
# include_directories( ${OpenCV_INCLUDE_DIRS} )

# ソースファイルを指定
set(SOURCES main.cpp ) 

# 実行可能ファイルをビルド
add_executable(Open_dnn ${SOURCES})

# OpenCVをリンク
target_link_libraries(Open_dnn 
                      PUBLIC
                      ${OpenCV_LIBS}
                      # Boostライブラリをリンク
                      ${Boost_LIBRARIES}
)

