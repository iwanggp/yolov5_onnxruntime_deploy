#ifndef DETECTOR_H
#define DETECTOR_H
#include <opencv2/opencv.hpp>
#include<onnxruntime_cxx_api.h>
#include<utility>
#include<iostream>
#include"utils.h"
using namespace std;
class Detector
{
public:
    Detector();
    void YOLODetector(const bool& isGPU,const cv::Size& inputSize);
    void getBestClassInfo(vector<float>::iterator it, const int& numClasses,
                                    float& bestConf, int& bestClassId);
    void preprocession(cv::Mat &image,float*& blob,vector<int64_t>& inputTensorShape);
    vector<Detection> postprocessing(const cv::Size& resizedImageShape,
                                     const cv::Size& originalImageShape,
                                     vector<Ort::Value>& outputTensors,
                                     const float& confThreshold,const float& iouThreshold);
    vector<Detection> detect(cv::Mat &image,const float& confThreshold,const float& iouThreshold);

private:
    string modelPath="E:/yolov5s-wang.onnx";
    Ort::Env env{nullptr};//构建Ort的环境，这是一个线性的固定写法
    Ort::SessionOptions sessionOptions{nullptr};//session选项
    Ort::Session session{nullptr};//Ort session管理器
    vector<const char*> inputNames;
    vector<const char*> outputNames;
    bool isDynamicInputShape;
    cv::Size2f inputImageShape;
    Utils utils;
};

#endif // DETECTOR_H
