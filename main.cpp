#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <vector>
#include <cassert>
#include <map>
#include <chrono>
#include "Config.h"
#include "Kernel.h"

using namespace std;
using namespace cv;

int main(int argc, char const *argv[])
{
    if ( argc != 4){
        cerr << "[ERROR] Please check argv! " << endl;
        return -1;
    }

    // Part 0：输入及预处理
    cv::Mat inRGB = imread(argv[1],CV_LOAD_IMAGE_UNCHANGED);
    cv::Mat inDepth = imread(argv[2],CV_LOAD_IMAGE_UNCHANGED);

    cv::FileStorage fsSettings(argv[3], cv::FileStorage::READ);
    if(!fsSettings.isOpened())
    {
        cerr << "[ERROR] Failed to open settings file at: " << argv[3] << endl;
        exit(-1);
    }
    Config conf(fsSettings);

    Kernel kernel(conf);

    std::chrono::system_clock::time_point  start, end;
    start = std::chrono::system_clock::now();

    cv::Mat result = kernel.FillInFast(inDepth);

    end = std::chrono::system_clock::now();
    double elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count();
    cout << "[INFO] Elapsed Times: " << elapsed << " ms." << endl;

    cv::imwrite("../result/showDepthMap.png", result);


    return 0;
}
