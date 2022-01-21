#include "utils.h"

Utils::Utils()
{

}
/**
 * @brief Utils::vectorProduct 生成vector
 * @param vector
 * @return
 */
size_t Utils::vectorProduct(const vector<int64_t> &vector)
{
    if(vector.empty())
        return 0;
    size_t product=1;
    for(const auto& element:vector)
        product*=element;
    return product;
}
/**
 * @brief Utils::charToWstring 将char转换为wstring格式
 * @param str
 * @return
 */
wstring Utils::charToWstring(const char *str)
{
    typedef codecvt_utf8<wchar_t> convert_type;
    wstring_convert<convert_type,wchar_t> converter;
    return converter.from_bytes(str);
}
/**
 * @brief Utils::loadNames 加载标签文件
 * @param path 标签路径
 * @return
 */
vector<string> Utils::loadNames(const string &path)
{
    vector<string> classNames;
    ifstream infile("E:/coco.names");//读取文件
    if(infile.good()){
        string line;
        while (getline(infile,line)) {
            if(line.back()=='\r')
                line.pop_back();
            classNames.emplace_back(line);
        }
        infile.close();
    }else {
        cerr<<"ERROR:Failed to access class name path: "<<path<<endl;
    }
    return classNames;
}
/**
 * @brief Utils::visualizeDetection 识别结果可视化
 * @param image
 * @param detections
 * @param classNames
 */
void Utils::visualizeDetection(cv::Mat &image, std::vector<Detection> &detections, const std::vector<string> &classNames)
{
    for (const Detection& detection : detections)
       {
           cv::rectangle(image, detection.box, cv::Scalar(229, 160, 21), 2);

           int x = detection.box.x;
           int y = detection.box.y;

           int conf = (int)std::round(detection.conf * 100);
           int classId = detection.classId;
           std::string label = classNames[classId] + " 0." + std::to_string(conf);

           int baseline = 0;
           cv::Size size = cv::getTextSize(label, cv::FONT_ITALIC, 0.8, 2, &baseline);
           cv::rectangle(image,
                         cv::Point(x, y - 25), cv::Point(x + size.width, y),
                         cv::Scalar(229, 160, 21), -1);

           cv::putText(image, label,
                       cv::Point(x, y - 3), cv::FONT_ITALIC,
                       0.8, cv::Scalar(255, 255, 255), 2);
       }
}
/**
 * @brief Utils::letterbox 信封的图片缩放与填充
 * @param image
 * @param outImage
 * @param newShape
 * @param color
 * @param auto_
 * @param scaleFill
 * @param scaleUp
 * @param stride
 */
void Utils::letterbox(const cv::Mat &image, cv::Mat &outImage, const cv::Size &newShape, const cv::Scalar &color=cv::Scalar(114,114,114), bool auto_=true, bool scaleFill=false, bool scaleUp=true, int stride=32)
{
    cv::Size shape=image.size();
    float r=min((float)newShape.height/(float)shape.height,(float)newShape.width/(float)shape.width);
    if(!scaleUp)
        r=min(r,1.0f);
    float ratio[2]{r,r};
    int newUnpad[2]{(int) round((float) shape.width*r),
                   (int) round((float)shape.height*r)};
    auto dw=(float)(newShape.width-newUnpad[0]);
    auto dh=(float)(newShape.height-newUnpad[1]);
    if(auto_){
        dw=(float)((int)dw%stride);
        dh=(float)((int)dh%stride);
    }
    else if(scaleFill){
        dw=0.0f;
        dh=0.0f;
        newUnpad[0]=newShape.width;
        newUnpad[1]=newShape.height;
        ratio[0]=(float)newShape.width/(float)shape.width;
        ratio[1]=(float)newShape.height/(float)shape.height;
    }
    dw/=2.0f;
    dh/=2.0f;
    if (shape.width != newUnpad[0] && shape.height != newUnpad[1])
    {
        cv::resize(image, outImage, cv::Size(newUnpad[0], newUnpad[1]));
    }

    int top = int(std::round(dh - 0.1f));
    int bottom = int(std::round(dh + 0.1f));
    int left = int(std::round(dw - 0.1f));
    int right = int(std::round(dw + 0.1f));
    cv::copyMakeBorder(outImage, outImage, top, bottom, left, right, cv::BORDER_CONSTANT, color);
}
/**
 * @brief Utils::scaleCoords 坐标还原
 * @param imageShape
 * @param box
 * @param imageOriginalShape
 */
void Utils::scaleCoords(const cv::Size &imageShape, cv::Rect &box, const cv::Size &imageOriginalShape)
{
    float gain=min((float)imageShape.height/(float)imageOriginalShape.height,
                   (float)imageShape.width/(float)imageOriginalShape.width);
    int pad[2]={(int)(((float)imageShape.width-(float)imageOriginalShape.width*gain)/2.0f),
               (int)(((float)imageShape.height-(float)imageOriginalShape.height*gain)/2.0f)};
    box.x=(int) round(((float)(box.x-pad[0])/gain));
    box.y=(int) round(((float)(box.y-pad[1])/gain));
    box.width=(int)round(((float)box.width/gain));
    box.height=(int)round(((float)box.height/gain));
}
