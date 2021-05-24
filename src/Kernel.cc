#include <string>
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>
#include <vector>
#include <cassert>
#include "Config.h"
#include "Kernel.h"

// #define DEBUG
#ifdef DEBUG
    #define CHECK_INFO(x) std::cout << "[DEBUG] " << x << std::endl;
    #define CHECK_INFO_2(x,y) std::cout << "[DEBUG] " << x << y << std::endl;
#else
    #define CHECK_INFO(x) //std::cout << x << std::endl;
    #define CHECK_INFO_2(x,y) //std::cout << "[DEBUG] " << x << y << std::endl;
#endif

Kernel::Kernel(Config &config): setting(config)
{
    std::cout << "[INFO]\tmax depth: " << setting.app.maxDepth << ";\n\tfill type: " << setting.app.fillMode
    << ";\n\textrapolate: " << setting.app.extrapolate << ";\n\tblur type: " << setting.app.blurType << std::endl;
}


cv::Mat Kernel::FillInFast(cv::Mat &depthMap){
    cv::Mat depth32F;
    depthMap.convertTo(depth32F, CV_32FC1, 1.0/256);

    #ifdef DEBUG
        cv::FileStorage fswrite("../result/test.xml", cv::FileStorage::WRITE);// 新建文件，覆盖掉已有文件
        fswrite << "depth32F" << depth32F;
        fswrite.release();
    #endif

    {
        size_t nr = depth32F.rows;
        size_t nc = depth32F.cols;
        if(depth32F.isContinuous())
        {
            nr = 1;
            nc = nc * depth32F.rows * depth32F.channels();
        }
        for(size_t i=0; i<nr; i++)
        {
            float* inData = depth32F.ptr<float>(i);
            for(size_t j=0; j< nc ; j++)
            {
                if (*inData > setting.app.minDepth)
                    *inData = setting.app.maxDepth - *inData;
                inData++;
            }
        }
    }


    morphologyEx(depth32F, depth32F, cv::MORPH_DILATE, DIAMOND_KERNEL_5);
    morphologyEx(depth32F, depth32F, cv::MORPH_CLOSE, FULL_KERNEL_5);

    cv::Mat dilated32F;
    morphologyEx(depth32F, dilated32F, cv::MORPH_DILATE, FULL_KERNEL_7);

    {
        size_t nr = depth32F.rows;
        size_t nc = depth32F.cols;
        if(depth32F.isContinuous() && dilated32F.isContinuous())
        {
            nr = 1;
            nc= nc*depth32F.rows * depth32F.channels();
        }
        for(size_t i=0; i<nr; i++)
        {
            float* outData = depth32F.ptr<float>(i);
            const float* inData = dilated32F.ptr<float>(i);
            for(size_t j=0; j< nc ; j++)
            {
                if (*outData < setting.app.minDepth){
                    *outData = *inData;
                }
                outData++;
                inData++;
            }
        }
    }


    if (setting.app.extrapolate){
        cv::Mat maskDepth = cv::Mat::zeros(depth32F.size(), CV_8UC1);
        cv::threshold(depth32F, maskDepth, setting.app.minDepth, setting.app.maxDepth, cv::THRESH_BINARY);

        #ifdef DEBUG
            cv::Mat maskDepthShow;
            maskDepth.convertTo(maskDepthShow, CV_8UC1);
            cv::imshow("maskDepthShow",maskDepthShow);
            cv::waitKey(0);
            cv::FileStorage fswrite("../result/maskDepth.xml", cv::FileStorage::WRITE);// 新建文件，覆盖掉已有文件
            fswrite << "maskDepth" << maskDepth;
            fswrite.release();
        #endif


        double min=0, max=0;
        cv::Point minLoc, maxLoc;
        std::vector<int> maxPosList;
        std::vector<float> maxDepthList;

        for (size_t x=0; x< maskDepth.cols; ++x){
            cv::Mat pixelCol = maskDepth.col(x);
            cv::minMaxLoc(pixelCol, &min, &max, &minLoc, &maxLoc);
            float tmpDepth = depth32F.at<float>(cv::Point(x,maxLoc.y));
            maxPosList.emplace_back(maxLoc.y);
            maxDepthList.emplace_back(tmpDepth);
            #ifdef DEBUG
                // std::cout << maxLoc.y << "\t";
                // std::cout << tmpDepth << "\t";
            #endif
        }

        for (size_t x=0; x< maskDepth.cols; ++x){
            for (size_t k=0; k< maxPosList[x]; ++k){
                depth32F.at<float>(cv::Point(x,k)) = maxDepthList[x];
            }
        }

        cv::Mat dilated;
        morphologyEx(depth32F, dilated, cv::MORPH_DILATE, FULL_KERNEL_31);

        size_t nr = depth32F.rows;
        size_t nc = depth32F.cols;
        if(depth32F.isContinuous() && dilated.isContinuous())
        {
            nr = 1;
            nc= nc*depth32F.rows * depth32F.channels();
        }
        for(size_t i=0; i<nr; i++)
        {
            float* outData = depth32F.ptr<float>(i);
            const float* inData = dilated.ptr<float>(i);
            for(size_t j=0; j< nc ; j++)
            {
                if (*outData < setting.app.minDepth){
                    *outData = *inData;
                }
                outData++;
                inData++;
            }
        }
    }


    cv::medianBlur(depth32F, depth32F, 5);

    cv::Mat blurred;
    if (setting.app.blurType == "bilateral"){
        bilateralFilter(depth32F, blurred, 5, 1.5, 2.0);
        depth32F = blurred.clone();
    }else if (setting.app.blurType == "gaussian"){
        cv::GaussianBlur(depth32F, blurred, cv::Size(5, 5), 0);
        {
            size_t nr = depth32F.rows;
            size_t nc = depth32F.cols;
            if(depth32F.isContinuous() && blurred.isContinuous())
            {
                nr = 1;
                nc= nc*depth32F.rows * depth32F.channels();
            }
            for(size_t i=0; i<nr; i++)
            {
                float* outData = depth32F.ptr<float>(i);
                const float* inData = blurred.ptr<float>(i);
                for(size_t j=0; j< nc ; j++)
                {
                    if (*outData > setting.app.minDepth){
                        *outData = *inData;
                    }
                    outData++;
                    inData++;
                }
            }
        }

    }

    cv::Mat result16U = cv::Mat::zeros(depth32F.size(), CV_16UC1);
    {
        size_t nr = depth32F.rows;
        size_t nc = depth32F.cols;
        if(depth32F.isContinuous() && result16U.isContinuous())
        {
            nr = 1;
            nc = nc*depth32F.rows * depth32F.channels();
        }
        for(size_t i=0; i<nr; i++)
        {
            float* inData   = depth32F.ptr<float>(i);
            ushort* outData = result16U.ptr<ushort>(i);
            for(size_t j=0; j< nc ; j++)
            {
                if (*inData > setting.app.minDepth){
                    *inData = setting.app.maxDepth - *inData;
                }

                *outData = static_cast<ushort>((*inData)*256.0);
                inData++;
                outData++;
            }
        }
    }

    return result16U;

}


// cv::Mat Kernel::FillInMultiscale(cv::Mat &depthMap){
//     cv::Mat depth32F;
//     depthMap.convertTo(depth32F, CV_32FC1, 1.0/256);

//     cv::Mat invertedDepthS1 = depth32F.clone();
//     cv::Mat validPixelsNear = cv::Mat::zeros(depth32F.size(), depth32F.type());
//     cv::Mat validPixelsMed  = cv::Mat::zeros(depth32F.size(), depth32F.type());
//     cv::Mat validPixelsFar  = cv::Mat::zeros(depth32F.size(), depth32F.type());
//     {
//         size_t nr = invertedDepthS1.rows;
//         size_t nc = invertedDepthS1.cols;
//         if(invertedDepthS1.isContinuous() && validPixelsNear.isContinuous() && validPixelsMed.isContinuous()
//             && validPixelsFar.isContinuous())
//         {
//             nr = 1;
//             nc = nc * invertedDepthS1.rows * invertedDepthS1.channels();
//         }
//         for(size_t i=0; i<nr; i++)
//         {
//             float* inData = invertedDepthS1.ptr<float>(i);
//             float* outData1 = validPixelsNear.ptr<float>(i);
//             float* outData2 = validPixelsMed.ptr<float>(i);
//             float* outData3 = validPixelsFar.ptr<float>(i);
//             for(size_t j=0; j< nc ; j++)
//             {
//                 if (*inData < setting.app.minDepth){
//                     *outData1 = 0;
//                     *outData2 = 0;
//                     *outData3 = 0;
//                 }else{
//                     if (*inData <= 15.0){
//                         *outData = 1;
//                         *outData2 = 1;
//                         *outData3 = 1;
//                     }else if (*inData <= 30.0){
//                         *outData2 = 1;
//                         *outData3 = 1;
//                     }else{
//                         *outData3 = 1;
//                     }
//                     *inData = setting.app.maxDepth - *inData;
//                 }
//                 inData++;
//                 outData1++;
//                 outData2++;
//                 outData3++;
//             }
//         }
//     }

//     cv::Mat dilationFar, dilationMed, dilationNear;
//     morphologyEx(invertedDepthS1.mul(validPixelsFar), dilationFar, cv::MORPH_DILATE, CROSS_KERNEL_3);
//     morphologyEx(invertedDepthS1.mul(validPixelsMed), dilationMed, cv::MORPH_DILATE, CROSS_KERNEL_5);
//     morphologyEx(invertedDepthS1.mul(validPixelsNear), dilationNear, cv::MORPH_DILATE, CROSS_KERNEL_7);


//     cv::Mat invertedDepthS2 = invertedDepthS1.clone();
//     {
//         size_t nr = invertedDepthS2.rows;
//         size_t nc = invertedDepthS2.cols;
//         if(invertedDepthS2.isContinuous() && dilationFar.isContinuous() && dilationMed.isContinuous()
//             && dilationNear.isContinuous())
//         {
//             nr = 1;
//             nc = nc * invertedDepthS2.rows * invertedDepthS2.channels();
//         }
//         for(size_t i=0; i<nr; i++)
//         {
//             float* outData = invertedDepthS2.ptr<float>(i);
//             const float* inData1 = dilationNear.ptr<float>(i);
//             const float* inData2 = dilationMed.ptr<float>(i);
//             const float* inData3 = dilationFar.ptr<float>(i);
//             for(size_t j=0; j< nc ; j++)
//             {
//                 if (*inData3 > setting.app.minDepth)
//                     *outData = *inData3;
//                 if (*inData2 > setting.app.minDepth)
//                     *outData = *inData2;
//                 if (*inData1 > setting.app.minDepth)
//                     *outData = *inData1;

//                 outData++;
//                 inData1++;
//                 inData2++;
//                 inData3++;
//             }
//         }
//     }

//     cv::Mat invertedDepthS3;
//     morphologyEx(invertedDepthS2, invertedDepthS3, cv::MORPH_CLOSE, FULL_KERNEL_5);

//     cv::Mat invertedDepthS4;
//     cv::medianBlur(invertedDepthS3, invertedDepthS4, 5);

//     {
//         size_t nr = invertedDepthS3.rows;
//         size_t nc = invertedDepthS3.cols;
//         if(invertedDepthS3.isContinuous() && invertedDepthS4.isContinuous())
//         {
//             nr = 1;
//             nc = nc*invertedDepthS3.rows * invertedDepthS3.channels();
//         }
//         for(size_t i=0; i<nr; i++)
//         {
//             float* outData = invertedDepthS4.ptr<float>(i);
//             float* inData  = invertedDepthS3.ptr<float>(i);
//             for(size_t j=0; j< nc ; j++)
//             {
//                 if (*outData < setting.app.minDepth){
//                     *outData = *inData;
//                 }
//                 outData++;
//                 inData++;
//             }
//         }
//     }

//     cv::Mat topMask = cv::Mat::zeros(invertedDepthS4.size(), CV_8UC1);

//     double min=0, max=0;
//     cv::Point minLoc, maxLoc;
//     for (size_t x=0; x< topMask.cols; ++x){
//         auto pixelCol = invertedDepthS4.col(x);
//         // cv::Mat pixelCol = invertedDepthS4.colRange(x, x+1).clone();
//         cv::minMaxLoc(pixelCol, &min, &max, &minLoc, &maxLoc);
//         for (size_t y=0; y<maxLoc; ++y){
//             topMask.at<uchar>(Point(x,y)) = 255;
//         }
//     }

//     cv::Mat dilated;
//     morphologyEx(invertedDepthS4, dilated, cv::MORPH_DILATE, FULL_KERNEL_9);

//     cv::Mat invertedDepthS5 = invertedDepthS4.clone();
//     {
//         size_t nr = invertedDepthS4.rows;
//         size_t nc = invertedDepthS4.cols;
//         if(invertedDepthS4.isContinuous() && dilated.isContinuous() && topMask.isContinuous())
//         {
//             nr = 1;
//             nc = nc * invertedDepthS4.rows * invertedDepthS4.channels();
//         }
//         for(size_t i=0; i<nr; i++)
//         {
//             float* outData = invertedDepthS5.ptr<float>(i);
//             const float* inData1 = dilated.ptr<float>(i);
//             const uchar* inData2 = topMask.ptr<uchar>(i);
//             for(size_t j=0; j< nc ; j++)
//             {
//                 if ((*inData1 < setting.app.minDepth) &&  *inData2==0 ){
//                     *outData = *inData1;
//                 }
//                 outData++;
//                 inData1++;
//                 inData2++;
//             }
//         }
//     }


//     cv::Mat topMask2 = cv::Mat::zeros(invertedDepthS5.size(), CV_8UC1);
//     cv::Mat invertedDepthS6 = invertedDepthS5.clone();
//     {
//         double min=0, max=0;
//         cv::Point minLoc, maxLoc;
//         for (size_t y=0; y< topMask2.rows; ++y){
//             auto pixelRow = invertedDepthS5.row(y);
//             if (setting.app.extrapolate){
//                 //pass
//             }
//             cv::minMaxLoc(pixelRow, &min, &max, &minLoc, &maxLoc);
//             for (size_t x=0; x<maxLoc; ++x){
//                 topMask2.at<uchar>(Point(x,y)) = 255;
//             }
//         }
//     }


//     return NULL;
// }