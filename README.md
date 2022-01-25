# yolov5-onnxruntime

本文目录如下：

- [ ] Python版onnxruntime部署Yolov5
- [ ] CPP版onnxruntime部署Yolov5
- [ ] TensorRT版onnxruntime部署Yolov5

### ONNXRuntime(ORT)

关于ONNX这里就不在赘述了，一句话总计**ONNX可以摆脱框架的依赖**。一般的ONNX部署深度学习模型可以分为**Opencv部署**和**ONNXRuntime部署**，但是Opencv毕竟只是第三方的库，对ONNX的支持并不是很友好，而且好多的模型还是不支持的，如果要使用还需要去修改模型的源码才可以。所以这时就体现出ONNXRuntime的优势了，ONNXRuntime几乎可以在不修改的源码的基础上进行部署。支持ONNX的runtime就是类似于JVM将统一的ONNX格式的模型包运行起来，包括对ONNX模型进行解读，优化，运行。它的整个架构就像Java的JVM机制一样。具体可以参考[onnxruntime.ai](https://onnxruntime.ai/)的具体介绍。

### 1 Python版用ONNXRuntime部署Yolov5

用Python部署yolov5模型几乎就是参照了源码的流程，。所以用python进行部署就会显得非常容易了，它主要如下的几个步骤：

1. 图片前处理阶段
2. 模型推理
3. 推理结果后处理

#### 1.1 图片预处理阶段

这部分是从Yolov3开始使用的图片预处理阶段，由于深度学习模型输入图片尺寸为正方形，而数据集中的图片一般为长方形，如果使用粗暴的使用resize会使图片失真，造成识别的误差。采用**letterbox**可以较好的解决这个问题。该方法可以保持图片的长宽比例，剩下的部分采用灰色填充(一般采用RGB色为114,114,114)。这部分实现代码如下：

```python
def letterbox(img, new_shape=(640, 640), auto=False, scaleFill=False, scaleUp=True):
    """
    python的信封图片缩放
    :param img: 原图
    :param new_shape: 缩放后的图片
    :param color: 填充的颜色
    :param auto: 是否为自动
    :param scaleFill: 填充
    :param scaleUp: 向上填充
    :return:
    """
    shape = img.shape[:2]  # current shape[height,width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleUp:
        r = min(r, 1.0)  # 确保不超过1
    ration = r, r  # width,height 缩放比例
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    if auto:
        dw, dh = np.mod(dw, 64), np.mod(dh, 64)
    elif scaleFill:
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ration = new_shape[1] / shape[1], new_shape[0] / shape[0]
    # 均分处理
    dw /= 2
    dh /= 2
    if shape[::-1] != new_unpad:
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))  # 添加边界
    return img
```

letterbox算法的主要思想是：首先将最长的边缩放到目标尺寸，然后将最短的边再用灰色去填充。其余的就是处理如何去填充了，比如使图片位于正中间等一系列操作。具体实现就是上面的代码了。这里还需要实现一下**坐标还原**操作，就是将letterbox的操作取一下逆运算，这样才能保证检测的坐标是正确的。具体代码如下：

```python
def scale_coords(img1_shape, coords, img0_shape, ratio_pad=None):
    """
    坐标还原
    :param img1_shape: 旧图像的尺寸
    :param coords: 坐标
    :param img0_shape:新图像的尺寸
    :param ratio_pad: 填充率
    :return:
    """
    if ratio_pad is None:  # 从img0_shape中计算
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain=old/new
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    coords[:, [0, 2]] -= pad[0]  # x padding
    coords[:, [1, 3]] -= pad[1]  # y padding
    coords[:, :4] /= gain
    clip_coords(coords, img0_shape)
    return coords
```



#### 1.2 模型推理

模型推理用Python来实现ONNXRuntime就非常简单了，差不多三行代码就可以实现。主要包括模型的读入、获得输入节点和获得输出节点以及最终的推理。该部分实现如下：

```python
sess=onnxruntime.InferenceSession(self.weights)#加载模型
#获得输入节点
self.input_name=sess.get_inputs()[0].name#获得输入节点
#获得输出节点
self.output_name=sess.get_outputs()[0].name
#执行推理获得原始的推理结果
pred=self.m.run(None,{self.input_name:img})[0]#执行推理
```

#### 1.3 推理结果后处理

获得模型的推理结果，下一步就是对结果进行后处理。后处理过程主要包括对检查结果中剥离出检测框、置信度以及检测类别。我们都知道YOLO输出的结果格式为**[centerX,centerY,width,height,BoxConfidence,all_class_confidence]**

前四个是目标检测框的输出分别表示目标框的中心点坐标，宽和高，第五个为检测框的输出置信度，最后一个各个类别的预测概率。这也就是为什么YOLO的输出的大小为**预测类别+5**的原因。还有一点很重要的是在对识别结果进行置信度过滤时好多博主**都错误的把目标框的置信度当做YOLO输出的置信度**，仔细阅读YOLO源码可知YOLO实际的输出置信度为**目标检测框的置信度与类别的乘积**，这一点很重要，如果不这样处理将会发现好多置信度为1的输出。后处理代码如下：

```python
    def postprocess(self, pred, conf_threshold, iou_threshold):
        """
        检测结果的后处理阶段，对检测结果进行后处理
        :param pred: 预测的原始结果
        :param conf_threshold: 置信度的阈值
        :param iou_threshold: iou的阈值
        :return: 预测的检测框，预测的置信度及预测的类别
        """
        boxes = []
        classIds = []
        confidences = []
        for detection in pred:
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID] * detection[4]  # 置信度为类别的概率和目标框概率值得乘积

            if confidence > conf_threshold:
                box = detection[0:4]  # 获取检测框的
                (centerX, centerY, width, height) = box.astype("int")
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))
                boxes.append([x, y, int(width), int(height)])
                classIds.append(classID)
                confidences.append(float(confidence))
        idxs = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, iou_threshold)  # 执行nms算法
        pred_boxes = []
        pred_confes = []
        pred_classes = []
        if len(idxs) > 0:
            for i in idxs.flatten():
                confidence = confidences[i]
                if confidence >= self.threshold:
                    pred_boxes.append(boxes[i])
                    pred_confes.append(confidence)
                    pred_classes.append(classIds[i])
        return pred_boxes, pred_confes, pred_classes
```

剩下的就是画框函数了，如果没问题将会得到如下的输出：

```
output node: output
output node: 345
output node: 403
output node: 461
['output', '345', '403', '461']
input name images-----output_name---output
input_shape: [1, 3, 640, 640]
```

说明程序已经可以正常运行了。

### 2 CPP版用ONNXRuntime部署Yolov5

用**cpp**语言来用ONNXRuntime部署YOLOv5就显得有点复杂了，其大致流程和前文的Python版本基本一致。复杂的就是ONNXRuntime这一部分的配置，不过ONNXRuntime基本上属于线性的结构一些写法都是固定的。只要按照特定的步骤就可以，它主要有三个主要的组件分别为：Ort::Env,Ort::SessionOptions()和OrtCUDAProviderOptions。所以的它的线性流程首先创建Ort的环境、设置SessionOptions()和设置cudaOption。

```cpp
    env=Ort::Env(OrtLoggingLevel::ORT_LOGGING_LEVEL_WARNING,"yolov5_ONNXRUNTIME");//创建yolov5运行环境
    sessionOptions=Ort::SessionOptions();//sessionOptions选择项
    sessionOptions.SetIntraOpNumThreads(1);   			          					      sessionOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
    OrtCUDAProviderOptions cudaOption;//cuda的选项配置
    sessionOptions.AppendExecutionProvider_CUDA(cudaOption);//加入cuda选项
    sessionOptions.AppendExecutionProvider_CUDA(cuda)
    vector<string> availableProviders=Ort::GetAvailableProviders();//获取Providers这是Ort的一般做法
    auto cudaAvailable=find(availableProviders.begin(),availableProviders.end(),"CUDAExecutionProvider");//遍历provides
    OrtCUDAProviderOptions cudaOption;
    //判断GPU是否可用
    if(isGPU&&(cudaAvailable==availableProviders.end())){
        cout<<"GPU is not supported by your ONNXRuntime build.Fallback to CPU."<<endl;
        cout<<"Inference device:CPU"<<endl;
    }
    else if(isGPU&&(cudaAvailable!=availableProviders.end())){
        cout<<"Inference Device :GPU"<<endl;
        sessionOptions.AppendExecutionProvider_CUDA(cudaOption);//增加CUDA选项
    }
    else{
        cout<<"Inference device:CPU "<<endl;
    }
#ifdef _WIN32
    wstring w_modelPath=utils.charToWstring(modelPath.c_str());//读模型
    session=Ort::Session(env,w_modelPath.c_str(),sessionOptions);
#else
    session=Ort::Session(env,modelPath.c_str(),sessionOptions);//初始化Session
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
```

初始完ONNXRuntime后下面的步骤几乎就与Python版本的一致了，图片的预处理、模型推理以及推理结果的后处理。

#### 2.1 图片预处理阶段

cpp版本的图片预处理同样包括了letterbox这种经典的图片预处理格式，不过后面需要将预处理后的图片再转换为浮点数有点复杂了，首先这里先看看cpp版的letterbox处理：

```cpp
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
```

整个流程和Python版本的差别不大，整体思想也几乎是一致的这里就不再仔细阐述了。下面重点是如何将预处理后的图片如何转换为浮点数，具体流程如下：

```cpp
 cv::Mat resizedImage,floatImage;
    cv::cvtColor(image,resizedImage,cv::COLOR_BGR2RGB);
    utils.letterbox(resizedImage,resizedImage,this->inputImageShape,cv::Scalar(114,114,114),this->isDynamicInputShape,false,true,32);
    inputTensorShape[2]=resizedImage.rows;#获得长宽
    inputTensorShape[3]=resizedImage.cols;
    resizedImage.convertTo(floatImage,CV_32FC3,1/255.0);#归一化
    blob=new float[floatImage.cols*floatImage.rows*floatImage.channels()];
    cv::Size floatImageSize{floatImage.cols,floatImage.rows};
    //hwc->chw转换
    vector<cv::Mat> chw(floatImage.channels());
    for(int i=0;i<floatImage.channels();++i)
    {
             							chw[i]=cv::Mat(floatImageSize,CV_32FC1,blob+i*floatImageSize.width*floatImageSize.height);
    }
    cv::split(floatImage,chw);#进行通道分离
```

#### 2.2 模型的推理

同样ONNNXRuntime的推理流程也是比较固定的，格式与流程也是比较固定的。首先是处理获得**inputTensor**、**申请内存**和**执行推理**。这一段代码如下所示：

```cpp
 //Ort的固定写法
    Ort::MemoryInfo memoryInfo=Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtArenaAllocator,OrtMemType::OrtMemTypeDefault);
    inputTensors.push_back(Ort::Value::CreateTensor<float>(memoryInfo,inputTensorValues.data(),inputTensorSize,inputTensorShape.data(),
                                                           inputTensorShape.size()));
    //执行推断
  	vector<Ort::Value> outputTensors=this->session.Run(Ort::RunOptions{nullptr},inputNames.data(),
                                                       inputTensors.data(),
                                                       1,
                                                       outputNames.data(),1);
```

整体的流程就是1申请Memory,2 加载inputTensor 3执行推断。这样就获得了模型的推理结果，下一步就是对推理结果的后处理操作。

#### 2.3 后处理阶段

整个后处理的过程大致和Python的过程一致这里就不再详细的写了，直接上代码注释：

```cpp
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
    auto* rawOutput=outputTensors[0].GetTensorData<float>();#获得输出的数据
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
```

