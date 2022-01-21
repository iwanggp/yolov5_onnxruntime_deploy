#ifndef UTILS_H
#define UTILS_H
#include<codecvt>
#include<fstream>
#include<opencv2/opencv.hpp>
#include<iostream>
using namespace std;
struct Detection
{
    cv::Rect box;//检测框
    float conf{};//置信度
    int classId{};//识别的类别
};

class Utils
{
public:
    Utils();
    size_t vectorProduct(const vector<int64_t>& vector);
    wstring charToWstring(const char* str);
    vector<string> loadNames(const string&path);//加载类别名称
    void visualizeDetection(cv::Mat& image, std::vector<Detection>& detections,
                                const std::vector<std::string>& classNames);
    void letterbox(const cv::Mat &image,cv::Mat& outImage,const cv::Size &newShape,const cv::Scalar &color,
                   bool auto_,bool scaleFill,bool scaleUp,int stride);//信封图片预处理
    void scaleCoords(const cv::Size& imageShape,cv::Rect& box,const cv::Size& imageOriginalShape);
    //模板函数
    template <typename T>
    T clip(const T& n,const T& lower,const T& upper);
};

#endif // UTILS_H
