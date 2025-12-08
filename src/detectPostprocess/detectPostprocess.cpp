#include "detectPostprocess.h"
#include "AclLiteApp.h"
#include <chrono>
#include "AclLiteUtils.h"
#include "Params.h"
#include "label.h"
#include <cstddef>
#include <iostream>
#include <iomanip>
#include <sstream>

using namespace std;

namespace
{
const uint32_t           kSleepTime = 500;
const vector<cv::Scalar> kColors{cv::Scalar(237, 149, 100),
                                 cv::Scalar(0, 215, 255),
                                 cv::Scalar(50, 205, 50),
                                 cv::Scalar(139, 85, 26)};
const float              kConfThresh = 0.25f;
const float              kNmsThresh = 0.45f;
const uint32_t           kNumClasses = 2;
typedef struct BoundBox
{
    float  x;
    float  y;
    float  width;
    float  height;
    float  score;
    size_t classIndex;
    size_t index;
} BoundBox;

bool sortScore(BoundBox box1, BoundBox box2) { return box1.score > box2.score; }
} // namespace

DetectPostprocessThread::DetectPostprocessThread(uint32_t      modelWidth,
                                                 uint32_t      modelHeight,
                                                 aclrtRunMode &runMode,
                                                 uint32_t      batch,
                                                 int           targetClassId)
    : modelWidth_(modelWidth),
      modelHeight_(modelHeight),
      runMode_(runMode),
      sendLastBatch_(false),
      batch_(batch),
      targetClassId_(targetClassId)
{
    if (targetClassId_ >= 0 && targetClassId_ >= static_cast<int>(kNumClasses))
    {
        ACLLITE_LOG_WARNING(
            "Configured target_class_id %d exceeds supported classes [%u), no "
            "detections will pass the filter",
            targetClassId_,
            kNumClasses);
    }
    else if (targetClassId_ >= 0)
    {
        ACLLITE_LOG_INFO("Enable target class filter: class_id=%d",
                         targetClassId_);
    }
}

DetectPostprocessThread::~DetectPostprocessThread() {}

AclLiteError DetectPostprocessThread::Init() { return ACLLITE_OK; }

AclLiteError DetectPostprocessThread::Process(int msgId, shared_ptr<void> data)
{
    auto start = std::chrono::high_resolution_clock::now();
    
    AclLiteError ret = ACLLITE_OK;
    switch (msgId)
    {
    case MSG_POSTPROC_DETECTDATA:
        InferOutputProcess(static_pointer_cast<DetectDataMsg>(data));
        MsgSend(static_pointer_cast<DetectDataMsg>(data));
        break;
    default:
        ACLLITE_LOG_INFO("Detect PostprocessThread thread ignore msg %d",
                         msgId);
        break;
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    if (msgId == MSG_POSTPROC_DETECTDATA) {
        static int logCount = 0;
        if (++logCount % 30 == 0) {
            ACLLITE_LOG_INFO("[%s] Process time: %ld ms", SelfInstanceName().c_str(), duration);
        }
    }

    return ret;
}

AclLiteError DetectPostprocessThread::InferOutputProcess(
    shared_ptr<DetectDataMsg> detectDataMsg)
{
    // OPTIMIZATION: Single batch copy to host (Step 2) instead of per-frame copy
    void *hostAllBuffer =
        CopyDataToHost(detectDataMsg->inferenceOutput[0].data.get(),
                       detectDataMsg->inferenceOutput[0].size,
                       runMode_,
                       MEMORY_NORMAL);
    if (hostAllBuffer == nullptr)
    {
        ACLLITE_LOG_ERROR("Copy inference output to host failed");
        return ACLLITE_ERROR_COPY_DATA;
    }
    
    float *hostBuff = static_cast<float *>(hostAllBuffer);
    size_t floatsPerFrame = (detectDataMsg->inferenceOutput[0].size / batch_) / sizeof(float);
    const uint32_t kNumChannels = 6; // cx, cy, w, h, conf_class0, conf_class1
    uint32_t numPredictionsPerFrame = (floatsPerFrame > 0) ? floatsPerFrame / kNumChannels : 0;
    
    for (size_t n = 0; n < detectDataMsg->decodedImg.size(); n++)
    {
        float *detectBuff = hostBuff + n * floatsPerFrame;

        // YOLOv7 model output: [batch, 6, 8400]
        // 6 channels: cx, cy, w, h, conf_class0, conf_class1

        // get srcImage width height
        int srcWidth = detectDataMsg->decodedImg[n].width;
        int srcHeight = detectDataMsg->decodedImg[n].height;

        // Calculate resize ratio (keep aspect ratio)
        float scaleWidth = (float)modelWidth_ / srcWidth;
        float scaleHeight = (float)modelHeight_ / srcHeight;
        float resizeRatio = min(scaleWidth, scaleHeight);
        
        // Calculate the actual resized dimensions
        float resizedWidth = srcWidth * resizeRatio;
        float resizedHeight = srcHeight * resizeRatio;
        
        // Calculate padding offset (image is centered in model input)
        float padLeft = (modelWidth_ - resizedWidth) / 2.0f;
        float padTop = (modelHeight_ - resizedHeight) / 2.0f;

        // filter boxes by confidence threshold
        // OPTIMIZATION: Pre-allocate vector (Step 5) to reduce allocations
        vector<BoundBox> boxes;
        boxes.reserve(std::min((size_t)1000, (size_t)numPredictionsPerFrame)); // reserve common detection count
        
        for (uint32_t i = 0; i < numPredictionsPerFrame; ++i)
        {
            // Extract center coordinates and dimensions (in model input image size)
            float cx = detectBuff[0 * numPredictionsPerFrame + i];
            float cy = detectBuff[1 * numPredictionsPerFrame + i];
            float w = detectBuff[2 * numPredictionsPerFrame + i];
            float h = detectBuff[3 * numPredictionsPerFrame + i];

            // Extract confidence scores for two classes
            float conf0 = detectBuff[4 * numPredictionsPerFrame + i];
            float conf1 = detectBuff[5 * numPredictionsPerFrame + i];

            // Select the class with higher confidence
            float score;
            uint32_t cls;
            if (conf0 > conf1)
            {
                score = conf0;
                cls = 0;
            }
            else
            {
                score = conf1;
                cls = 1;
            }

            // Filter by confidence threshold
            if (score <= kConfThresh)
                continue;
            
            // Optional class-id filtering from config
            if (targetClassId_ >= 0 &&
                cls != static_cast<uint32_t>(targetClassId_))
            {
                continue;
            }

            // Convert center coordinates to corner coordinates in model input size
            float x1 = cx - w / 2.0f;
            float y1 = cy - h / 2.0f;
            float x2 = cx + w / 2.0f;
            float y2 = cy + h / 2.0f;

            // Remove padding offset first, then scale to original image size
            x1 = (x1 - padLeft) / resizeRatio;
            y1 = (y1 - padTop) / resizeRatio;
            x2 = (x2 - padLeft) / resizeRatio;
            y2 = (y2 - padTop) / resizeRatio;

            // Clip coordinates to valid range
            x1 = max(0.0f, min(x1, (float)(srcWidth - 1)));
            y1 = max(0.0f, min(y1, (float)(srcHeight - 1)));
            x2 = max(0.0f, min(x2, (float)(srcWidth - 1)));
            y2 = max(0.0f, min(y2, (float)(srcHeight - 1)));

            // Convert to center coordinates and size for BoundBox
            BoundBox box;
            box.x = (x1 + x2) / 2.0f;
            box.y = (y1 + y2) / 2.0f;
            box.width = x2 - x1;
            box.height = y2 - y1;
            box.score = score;
            box.classIndex = cls;
            box.index = i;
            
            if (cls < kNumClasses)
            {
                boxes.push_back(box);
            }
        }

        // filter boxes by NMS
        // OPTIMIZATION: Index-based NMS (Step 4) to avoid expensive vector::erase operations
        vector<BoundBox> result;
        result.reserve(boxes.size()); // pre-allocate (Step 5)
        
        if (!boxes.empty())
        {
            std::sort(boxes.begin(), boxes.end(), sortScore);
            
            // Use suppression flags instead of erasing elements
            vector<bool> suppressed(boxes.size(), false);
            
            for (size_t i = 0; i < boxes.size(); ++i)
            {
                if (suppressed[i]) continue;
                
                result.push_back(boxes[i]);
                
                // Suppress boxes with high IoU
                for (size_t j = i + 1; j < boxes.size(); ++j)
                {
                    if (suppressed[j]) continue;
                    
                    // Calculate IoU between boxes[i] and boxes[j]
                    float xLeft = max(boxes[i].x, boxes[j].x);
                    float yTop = max(boxes[i].y, boxes[j].y);
                    float xRight = min(boxes[i].x + boxes[i].width,
                                       boxes[j].x + boxes[j].width);
                    float yBottom = min(boxes[i].y + boxes[i].height,
                                        boxes[j].y + boxes[j].height);
                    float width = max(0.0f, xRight - xLeft);
                    float hight = max(0.0f, yBottom - yTop);
                    float area = width * hight;
                    float iou =
                        area / (boxes[i].width * boxes[i].height +
                                boxes[j].width * boxes[j].height - area);
                    
                    // Suppress box if IoU exceeds threshold
                    if (iou > kNmsThresh)
                    {
                        suppressed[j] = true;
                    }
                }
            }
        }

        // opencv draw label params
        int half = 2;

        cv::Point leftTopPoint;     // left top
        cv::Point rightBottomPoint; // right bottom
        string    className;        // yolo detect output

        // calculate framenum
        int frameCnt = (detectDataMsg->msgNum) * batch_ + n + 1;

        stringstream sstream;
        sstream.str("");
        sstream << "Channel-" << detectDataMsg->channelId << "-Frame-"
                << to_string(frameCnt) << "-result: ";

        string textHead = "";
        sstream >> textHead;
        string textMid = "[";
        
        // Pre-allocate detection vectors (Step 5)
        detectDataMsg->detections.reserve(detectDataMsg->detections.size() + result.size());
        
        // ============ 选择最佳检测目标(最接近画面中心且置信度大于阈值) ============
        DetectionOBB bestDetection;
        bool hasBestDetection = false;
        float minDistanceToCenter = std::numeric_limits<float>::max();
        float imageCenterX = srcWidth / 2.0f;
        float imageCenterY = srcHeight / 2.0f;
        
        for (size_t i = 0; i < result.size(); ++i)
        {
            // Store structured detection results in DetectDataMsg.detections for tracking
            DetectionOBB det;
            float x1_det = result[i].x - result[i].width / half;
            float y1_det = result[i].y - result[i].height / half;
            float x2_det = result[i].x + result[i].width / half;
            float y2_det = result[i].y + result[i].height / half;
            det.x0 = x1_det;
            det.y0 = y1_det;
            det.x1 = x2_det;
            det.y1 = y2_det;
            det.score = result[i].score;
            det.class_id = static_cast<int>(result[i].classIndex);
            detectDataMsg->detections.push_back(det);
            
            // 计算目标中心到画面中心的距离
            float detCenterX = result[i].x;
            float detCenterY = result[i].y;
            float distanceToCenter = std::sqrt(
                std::pow(detCenterX - imageCenterX, 2) + 
                std::pow(detCenterY - imageCenterY, 2)
            );
            
            // 选择距离中心最近的目标作为最佳检测
            if (distanceToCenter < minDistanceToCenter)
            {
                minDistanceToCenter = distanceToCenter;
                bestDetection = det;
                hasBestDetection = true;
            }

            leftTopPoint.x = result[i].x - result[i].width / half;
            leftTopPoint.y = result[i].y - result[i].height / half;
            rightBottomPoint.x = result[i].x + result[i].width / half;
            rightBottomPoint.y = result[i].y + result[i].height / half;
            
            // Format confidence score to 2 decimal places
            std::stringstream ss;
            ss << std::fixed << std::setprecision(2) << result[i].score;
            std::string scoreStr = ss.str();
            
            className = label[result[i].classIndex] + ":" + scoreStr;
            
            textMid = textMid + className + " ";
        }
        
        // 如果找到最佳检测目标，标记为跟踪初始化目标
        if (hasBestDetection)
        {
            // 将最佳检测目标放在detections数组的首位(方便跟踪模块获取)
            if (!detectDataMsg->detections.empty())
            {
                // 将最佳目标与第一个元素交换
                for (size_t i = 0; i < detectDataMsg->detections.size(); ++i)
                {
                    if (std::abs(detectDataMsg->detections[i].x0 - bestDetection.x0) < 1.0f &&
                        std::abs(detectDataMsg->detections[i].y0 - bestDetection.y0) < 1.0f)
                    {
                        std::swap(detectDataMsg->detections[0], detectDataMsg->detections[i]);
                        break;
                    }
                }
            }
        }
        
        string textPrint = textHead + textMid + "]";
        detectDataMsg->textPrint.push_back(textPrint);
    }
    
    // OPTIMIZATION: Single cleanup for entire batch (Step 1) - use delete[] instead of free
    delete[] static_cast<uint8_t*>(hostAllBuffer);
    
    return ACLLITE_OK;
}

AclLiteError
DetectPostprocessThread::MsgSend(shared_ptr<DetectDataMsg> detectDataMsg)
{
    if (!sendLastBatch_)
    {
        int targetThreadId = detectDataMsg->dataOutputThreadId;
        int targetMsgId = MSG_OUTPUT_FRAME;
        if (detectDataMsg->trackThreadId != INVALID_INSTANCE_ID)
        {
            targetThreadId = detectDataMsg->trackThreadId;
            targetMsgId = MSG_TRACK_DATA;
        }
        while (1)
        {
            AclLiteError ret = SendMessage(targetThreadId,
                                           targetMsgId,
                                           detectDataMsg);
            if (ret == ACLLITE_ERROR_ENQUEUE)
            {
                usleep(kSleepTime);
                continue;
            }
            else if (ret == ACLLITE_OK)
            {
                break;
            }
            else
            {
                ACLLITE_LOG_ERROR("Send read frame message failed, error %d",
                                  ret);
                return ret;
            }
        }
    }
    if (detectDataMsg->isLastFrame && sendLastBatch_)
    {
        while (1)
        {
            AclLiteError ret = SendMessage(detectDataMsg->dataOutputThreadId,
                                           MSG_ENCODE_FINISH,
                                           detectDataMsg);
            if (ret == ACLLITE_ERROR_ENQUEUE)
            {
                usleep(kSleepTime);
                continue;
            }
            else if (ret == ACLLITE_OK)
            {
                break;
            }
            else
            {
                ACLLITE_LOG_ERROR("Send read frame message failed, error %d",
                                  ret);
                return ret;
            }
        }
    }
    if (detectDataMsg->isLastFrame && !sendLastBatch_)
    {
        while (1)
        {
            AclLiteError ret = SendMessage(detectDataMsg->dataOutputThreadId,
                                           MSG_ENCODE_FINISH,
                                           detectDataMsg);
            if (ret == ACLLITE_ERROR_ENQUEUE)
            {
                usleep(kSleepTime);
                continue;
            }
            else if (ret == ACLLITE_OK)
            {
                break;
            }
            else
            {
                ACLLITE_LOG_ERROR("Send read frame message failed, error %d",
                                  ret);
                return ret;
            }
        }
        sendLastBatch_ = true;
    }

    return ACLLITE_OK;
}
