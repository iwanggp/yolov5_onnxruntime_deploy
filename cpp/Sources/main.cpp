#include "mainwindow.h"
#include <QApplication>
#include "detector.h"
using namespace std;
int main(int argc, char *argv[])
{
    Detector detector;
    bool isGPU=true;
    Utils utils;
    const vector<string> classNames=utils.loadNames("E:/coco.names");
    if(classNames.empty()){
        cerr<<"Error: Empty class names file"<<endl;
        return -1;
    }
    cv::Mat image;
    vector<Detection> result;
    try{
        detector.YOLODetector(isGPU,cv::Size(640,640));
        cout<<"Model was initialized......"<<endl;
        image=cv::imread("E:/plates/nan.jpg");
        result=detector.detect(image,0.4,0.4);

    }catch(const exception& e){
        cerr<<e.what()<<endl;
        return -1;
    }
    utils.visualizeDetection(image,result,classNames);
    cv::imshow("resul4444t",image);
    cv::waitKey(0);
    return 0;
}
