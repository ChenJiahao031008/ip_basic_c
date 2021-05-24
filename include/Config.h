/*
 * @Author: your name
 * @Date: 2021-03-26 11:25:51
 * @LastEditTime: 2021-05-03 21:11:28
 * @LastEditors: Please set LastEditors
 * @Description: In User Settings Edit
 * @FilePath: /极线可视化/include/config.h
 */
#ifndef CONFIG_H
#define CONFIG_H

#include <string>
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>
#include <vector>
#include <cassert>

class Config
{
public:
    struct AppSettings
    {
        std::string datasetPath;
        std::string fillMode;
        std::string blurType;
        int extrapolate;
        int resize;
        float maxDepth;
        float minDepth;
    };


public:
    cv::FileStorage SettingsFile;
    AppSettings app;


public:
    Config(cv::FileStorage &fsSettings):SettingsFile(fsSettings)
    {
        AppSettingsInit();
    };

    void AppSettingsInit(){
        app.datasetPath = static_cast<std::string>(SettingsFile["DatasetDir"]);
        app.fillMode = static_cast<std::string>(SettingsFile["FillType"]);
        app.blurType = static_cast<std::string>(SettingsFile["BlurType"]);
        app.extrapolate = SettingsFile["Extrapolate"];
        app.resize = SettingsFile["Resize"];
        app.maxDepth = SettingsFile["maxDepth"];
        app.minDepth = SettingsFile["minDepth"];
    };
};

#endif //CONFIG_H

