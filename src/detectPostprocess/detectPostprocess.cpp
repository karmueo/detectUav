#include "detectPostprocess.h"
#include "AclLiteApp.h"
#include <chrono>
#include "AclLiteUtils.h"
#include "Params.h"
#include "label.h"
#include <cstddef>
#include <iostream>

using namespace std;

namespace
{
const uint32_t           kSleepTime = 500;
const double             kFountScale = 0.5;
const cv::Scalar         kFountColor(0, 0, 255);
const uint32_t           kLabelOffset = 11;
const uint32_t           kLineSolid = 2;
const vector<cv::Scalar> kColors{cv::Scalar(237, 149, 100),
                                 cv::Scalar(0, 215, 255),
                                 cv::Scalar(50, 205, 50),
                                 cv::Scalar(139, 85, 26)};
const uint32_t           kNumPredictions = 8400;
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
                                                 uint32_t      batch)
    : modelWidth_(modelWidth),
      modelHeight_(modelHeight),
      runMode_(runMode),
      sendLastBatch_(false),
      batch_(batch)
{
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
            ACLLITE_LOG_INFO("[DetectPostprocessThread] Process time: %ld ms", duration);
        }
    }

    return ret;
}

AclLiteError DetectPostprocessThread::InferOutputProcess(
    shared_ptr<DetectDataMsg> detectDataMsg)
{
    size_t pos = 0;
    for (size_t n = 0; n < detectDataMsg->decodedImg.size(); n++)
    {
        void *dataBuffer =
            CopyDataToHost((char*)detectDataMsg->inferenceOutput[0].data.get() + pos,
                           detectDataMsg->inferenceOutput[0].size / batch_,
                           runMode_,
                           MEMORY_NORMAL);
        if (dataBuffer == nullptr)
        {
            ACLLITE_LOG_ERROR("Copy inference output to host failed");
            return ACLLITE_ERROR_COPY_DATA;
        }
        pos = pos + detectDataMsg->inferenceOutput[0].size / batch_;
        float *detectBuff = static_cast<float *>(dataBuffer);

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
        vector<BoundBox> boxes;
        
        for (uint32_t i = 0; i < kNumPredictions; ++i)
        {
            // Extract center coordinates and dimensions (in model input image size)
            float cx = detectBuff[0 * kNumPredictions + i];
            float cy = detectBuff[1 * kNumPredictions + i];
            float w = detectBuff[2 * kNumPredictions + i];
            float h = detectBuff[3 * kNumPredictions + i];

            // Extract confidence scores for two classes
            float conf0 = detectBuff[4 * kNumPredictions + i];
            float conf1 = detectBuff[5 * kNumPredictions + i];

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
        vector<BoundBox> result;
        result.clear();
        int32_t maxLength = srcWidth > srcHeight ? srcWidth : srcHeight;
        std::sort(boxes.begin(), boxes.end(), sortScore);
        BoundBox boxMax;
        BoundBox boxCompare;
        while (boxes.size() != 0)
        {
            size_t index = 1;
            result.push_back(boxes[0]);
            while (boxes.size() > index)
            {
                boxMax.score = boxes[0].score;
                boxMax.classIndex = boxes[0].classIndex;
                boxMax.index = boxes[0].index;

                // translate point by maxLength * boxes[0].classIndex to
                // avoid bumping into two boxes of different classes
                boxMax.x = boxes[0].x + maxLength * boxes[0].classIndex;
                boxMax.y = boxes[0].y + maxLength * boxes[0].classIndex;
                boxMax.width = boxes[0].width;
                boxMax.height = boxes[0].height;

                boxCompare.score = boxes[index].score;
                boxCompare.classIndex = boxes[index].classIndex;
                boxCompare.index = boxes[index].index;

                // translate point by maxLength * boxes[0].classIndex to
                // avoid bumping into two boxes of different classes
                boxCompare.x =
                    boxes[index].x + boxes[index].classIndex * maxLength;
                boxCompare.y =
                    boxes[index].y + boxes[index].classIndex * maxLength;
                boxCompare.width = boxes[index].width;
                boxCompare.height = boxes[index].height;

                // the overlapping part of the two boxes
                float xLeft = max(boxMax.x, boxCompare.x);
                float yTop = max(boxMax.y, boxCompare.y);
                float xRight = min(boxMax.x + boxMax.width,
                                   boxCompare.x + boxCompare.width);
                float yBottom = min(boxMax.y + boxMax.height,
                                    boxCompare.y + boxCompare.height);
                float width = max(0.0f, xRight - xLeft);
                float hight = max(0.0f, yBottom - yTop);
                float area = width * hight;
                float iou =
                    area / (boxMax.width * boxMax.height +
                            boxCompare.width * boxCompare.height - area);

                // filter boxes by NMS threshold
                if (iou > kNmsThresh)
                {
                    boxes.erase(boxes.begin() + index);
                    continue;
                }
                ++index;
            }
            boxes.erase(boxes.begin());
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
        for (size_t i = 0; i < result.size(); ++i)
        {
            leftTopPoint.x = result[i].x - result[i].width / half;
            leftTopPoint.y = result[i].y - result[i].height / half;
            rightBottomPoint.x = result[i].x + result[i].width / half;
            rightBottomPoint.y = result[i].y + result[i].height / half;
            className =
                label[result[i].classIndex] + ":" + to_string(result[i].score);
            cv::rectangle(detectDataMsg->frame[n],
                          leftTopPoint,
                          rightBottomPoint,
                          kColors[i % kColors.size()],
                          kLineSolid);
            cv::putText(
                detectDataMsg->frame[n],
                className,
                cv::Point(leftTopPoint.x, leftTopPoint.y + kLabelOffset),
                cv::FONT_HERSHEY_COMPLEX,
                kFountScale,
                kFountColor);
            textMid = textMid + className + " ";
        }
        string textPrint = textHead + textMid + "]";
        detectDataMsg->textPrint.push_back(textPrint);
        free(detectBuff);
        detectBuff = nullptr;
    }
    return ACLLITE_OK;
}

AclLiteError
DetectPostprocessThread::MsgSend(shared_ptr<DetectDataMsg> detectDataMsg)
{
    if (!sendLastBatch_)
    {
        while (1)
        {
            AclLiteError ret = SendMessage(detectDataMsg->dataOutputThreadId,
                                           MSG_OUTPUT_FRAME,
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