#include "detector.h"
#include "utils.h"
Detector::Detector()
{

}
/**
 * @brief Detector::YOLODetector yolo检测进行onnxruntime的初始化
 * @param isGPU
 * @param inputSize
 */
void Detector::YOLODetector(const bool &isGPU=true, const cv::Size &inputSize=cv::Size(640,640))
{
    env=Ort::Env(OrtLoggingLevel::ORT_LOGGING_LEVEL_WARNING,"yolov5_ONNXRUNTIME");//创建yolov5运行环境
    sessionOptions=Ort::SessionOptions();//sessionOptions选择项
    sessionOptions.SetIntraOpNumThreads(1);
    sessionOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
    OrtCUDAProviderOptions cudaOption;
//    sessionOptions.AppendExecutionProvider_CUDA(cudaOption);
//    sessionOptions.AppendExecutionProvider_CUDA(cuda
//    vector<string> availableProviders=Ort::GetAvailableProviders();//获取Providers这是Ort的一般做法
//    auto cudaAvailable=find(availableProviders.begin(),availableProviders.end(),"CUDAExecutionProvider");//遍历provides
//    OrtCUDAProviderOptions cudaOption;
//    //判断GPU是否可用
//    if(isGPU&&(cudaAvailable==availableProviders.end())){
//        cout<<"GPU is not supported by your ONNXRuntime build.Fallback to CPU."<<endl;
//        cout<<"Inference device:CPU"<<endl;
//    }
//    else if(isGPU&&(cudaAvailable!=availableProviders.end())){
//        cout<<"Inference Device :GPU"<<endl;
//        sessionOptions.AppendExecutionProvider_CUDA(cudaOption);//增加CUDA选项
//    }
//    else{
//        cout<<"Inference device:CPU "<<endl;
//    }
#ifdef _WIN32
    wstring w_modelPath=utils.charToWstring(modelPath.c_str());
    session=Ort::Session(env,w_modelPath.c_str(),sessionOptions);
#else
    session=Ort::Session(env,modelPath.c_str(),sessionOptions);
#endif
    Ort::AllocatorWithDefaultOptions allocator;//设置allocator分配
    Ort::TypeInfo inputTypeInfo=session.GetInputTypeInfo(0);//获取输入的类型信息
    vector<int64_t> inputTensorShape=inputTypeInfo.GetTensorTypeAndShapeInfo().GetShape();//获取输入的tensorshape
    this->isDynamicInputShape=false;
    //如下也是固定写法判断高和宽是不是固定的
    if(inputTensorShape[2]==-1&&inputTensorShape[3]==-1)
    {
        cout<<"Dynamic input shape"<<endl;
        this->isDynamicInputShape=true;
    }
    for(auto shape:inputTensorShape){
        cout<<"Input shape: "<<shape<<endl;
    }
    //将输入和输出的节点名字添加进来
    inputNames.push_back(session.GetInputName(0,allocator));
    outputNames.push_back(session.GetOutputName(0,allocator));
    cout<<"Input name :"<<inputNames[0]<<endl;
    cout<<"Output name 1:"<<outputNames[0]<<endl;
    this->inputImageShape=cv::Size2f(inputSize);
}

void Detector::getBestClassInfo(vector<float>::iterator it, const int &numClasses, float &bestConf, int &bestClassId)
{
    bestClassId = 5;
    bestConf = 0;
    for (int i = 5; i < numClasses + 5; i++)
    {
        if (it[i] > bestConf)
        {
            bestConf = it[i];
            bestClassId = i - 5;
        }
    }
}
/**
 * @brief Detector::preprocession 图片预处理过程
 * @param image
 * @param blob
 * @param inputTensorShape
 */
void Detector::preprocession(cv::Mat &image, float *&blob, vector<int64_t> &inputTensorShape)
{
    cv::Mat resizedImage,floatImage;
    cv::cvtColor(image,resizedImage,cv::COLOR_BGR2RGB);
    utils.letterbox(resizedImage,resizedImage,this->inputImageShape,cv::Scalar(114,114,114),this->isDynamicInputShape,false,true,32);
    inputTensorShape[2]=resizedImage.rows;
    inputTensorShape[3]=resizedImage.cols;

    resizedImage.convertTo(floatImage,CV_32FC3,1/255.0);
    blob=new float[floatImage.cols*floatImage.rows*floatImage.channels()];
    cv::Size floatImageSize{floatImage.cols,floatImage.rows};
    //hwc->chw转换
    vector<cv::Mat> chw(floatImage.channels());
    for(int i=0;i<floatImage.channels();++i)
    {
        chw[i]=cv::Mat(floatImageSize,CV_32FC1,blob+i*floatImageSize.width*floatImageSize.height);
    }
    cv::split(floatImage,chw);
}
/**
 * @brief Detector::postprocessing 识别结果后处理阶段
 * @param resizedImageShape 缩放后的图片
 * @param originalImageShape 原始图片
 * @param outputTensors 输出的Tensors
 * @param confThreshold 置信度
 * @param iouThreshold iou的值
 * @return
 */
vector<Detection> Detector::postprocessing(const cv::Size &resizedImageShape, const cv::Size &originalImageShape, vector<Ort::Value> &outputTensors, const float &confThreshold, const float &iouThreshold)
{
    vector<cv::Rect> boxes;
    vector<float> confs;
    vector<int> classIds;
    auto* rawOutput=outputTensors[0].GetTensorData<float>();
    vector<int64_t> outputShape=outputTensors[0].GetTensorTypeAndShapeInfo().GetShape();
    size_t count=outputTensors[0].GetTensorTypeAndShapeInfo().GetElementCount();
    vector<float> output(rawOutput,rawOutput+count);
    int numClasses=(int)outputShape[2]-5;
    int elementsInBatch=(int)(outputShape[1]*outputShape[2]);
    //only for batch size=1
    for(auto it=output.begin();it!=output.begin()+elementsInBatch;it+=outputShape[2])
    {
        float clsConf=it[4];
        if(clsConf>confThreshold)
        {
            int centerX=(int)(it[0]);
            int centerY=(int)(it[1]);
            int width=(int)(it[2]);
            int height=(int)(it[3]);
            int left=centerX-width/2;
            int top=centerY-height/2;
            float objConf;
            int classId;
            this->getBestClassInfo(it,numClasses,objConf,classId);
            float confidence=clsConf*objConf;
            boxes.emplace_back(left,top,width,height);
            confs.emplace_back(confidence);
            classIds.emplace_back(classId);
        }
    }
    vector<int> indices;
    cv::dnn::NMSBoxes(boxes,confs,confThreshold,iouThreshold,indices);
    vector<Detection> detections;

    for(int idx:indices){
        Detection det;
        det.box=cv::Rect(boxes[idx]);
        //将检测后的坐标还原回去
        utils.scaleCoords(resizedImageShape,det.box,originalImageShape);
        det.conf=confs[idx];
        det.classId=classIds[idx];
        cout<<"the ..........."<<classIds[idx];
        detections.emplace_back(det);
    }
    return detections;
}
/**
 * @brief Detector::detect 检测的主代码
 * @param image 图片
 * @param confThreshold 置信度
 * @param iouThreshold iou阈值
 * @return
 */
vector<Detection> Detector::detect(cv::Mat &image, const float &confThreshold=0.4, const float &iouThreshold=0.45)
{
    float *blob=nullptr;
    vector<int64_t> inputTensorShape{1,3,-1,-1};
    this->preprocession(image,blob,inputTensorShape);
    size_t inputTensorSize=utils.vectorProduct(inputTensorShape);
    vector<float> inputTensorValues(blob,blob+inputTensorSize);
    vector<Ort::Value> inputTensors;
    //Ort的固定写法
    Ort::MemoryInfo memoryInfo=Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtArenaAllocator,OrtMemType::OrtMemTypeDefault);
    inputTensors.push_back(Ort::Value::CreateTensor<float>(memoryInfo,inputTensorValues.data(),inputTensorSize,inputTensorShape.data(),
                                                           inputTensorShape.size()));
    //执行推断
    vector<Ort::Value> outputTensors=this->session.Run(Ort::RunOptions{nullptr},inputNames.data(),
                                                       inputTensors.data(),
                                                       1,
                                                       outputNames.data(),1);
    cv::Size resizedShape=cv::Size((int)inputTensorShape[3],(int)inputTensorShape[2]);
    vector<Detection> result=this->postprocessing(resizedShape,image.size(),outputTensors,confThreshold,iouThreshold);
    delete[] blob;
    return result;
}

