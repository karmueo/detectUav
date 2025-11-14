#include "pushrtspthread.h"
#include "AclLiteApp.h"
#include <chrono>
#include <cstddef>
using namespace cv;
using namespace std;
namespace
{
uint32_t kResizeWidth = 960;   // 方案1:降低分辨率以提升性能 (原1280)
uint32_t kResizeHeight = 540;  // 方案1:降低分辨率以提升性能 (原720)
uint32_t kBgrMultiplier = 3;
} // namespace

PushRtspThread::PushRtspThread(std::string rtspUrl)
{
    g_rtspUrl = rtspUrl;
    ACLLITE_LOG_INFO("PushRtspThread URL : %s", g_rtspUrl.c_str());
}

PushRtspThread::~PushRtspThread() { g_picToRtsp.FlushEncoder(); }

AclLiteError PushRtspThread::Init()
{
    g_frameSeq = 0;
    XInitThreads();
    
    // 获取当前ACL context用于硬件编码器
    aclrtContext context = nullptr;
    aclError aclRet = aclrtGetCurrentContext(&context);
    if (aclRet != ACL_SUCCESS)
    {
        ACLLITE_LOG_ERROR("Failed to get ACL context, error: %d", aclRet);
        context = nullptr; // VideoWriter会自动获取
    }
    
    AclLiteError ret = g_picToRtsp.AvInit(kResizeWidth, kResizeHeight, g_rtspUrl, context);
    if (ret != ACLLITE_OK)
    {
        ACLLITE_LOG_ERROR("AvInit rtsp failed");
        return ACLLITE_ERROR;
    }
    g_picToRtsp.BgrDataInint();
    return ACLLITE_OK;
}

AclLiteError PushRtspThread::Process(int msgId, std::shared_ptr<void> msgData)
{
    auto start = std::chrono::high_resolution_clock::now();
    
    switch (msgId)
    {
    case MSG_RTSP_DISPLAY:
        DisplayMsgProcess(static_pointer_cast<DetectDataMsg>(msgData));
        break;
    case MSG_ENCODE_FINISH:
        SendMessage(g_MainThreadId, MSG_APP_EXIT, nullptr);
        break;
    default:
        ACLLITE_LOG_INFO("Present agent display thread ignore msg %d", msgId);
        break;
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    if (msgId == MSG_RTSP_DISPLAY) {
        static int logCount = 0;
        if (++logCount % 30 == 0) {
            ACLLITE_LOG_INFO("[PushRtspThread] Process time: %ld ms", duration);
        }
    }
    
    return ACLLITE_OK;
}

AclLiteError
PushRtspThread::DisplayMsgProcess(std::shared_ptr<DetectDataMsg> detectDataMsg)
{
    static int frameCount = 0;
    frameCount++;
    
    if (frameCount == 1 || frameCount % 30 == 0) {
        ACLLITE_LOG_INFO("Processing frame %d, frames in batch: %zu, isLastFrame: %d",
                         frameCount, detectDataMsg->frame.size(), detectDataMsg->isLastFrame);
    }
    
    if (detectDataMsg->isLastFrame)
    {
        if (av_log_get_level() != AV_LOG_ERROR)
        {
            av_log_set_level(AV_LOG_ERROR);
        }
        for (int i = 0; i < detectDataMsg->frame.size(); i++)
        {
            // 数据已在DataOutput中resize,直接使用
            cv::Mat &frame = detectDataMsg->frame[i];
            g_picToRtsp.BgrDataToRtsp(frame.data,
                                      frame.cols * frame.rows * kBgrMultiplier,
                                      g_frameSeq);  // 使用当前序号
        }
        g_frameSeq++;  // 推流完成后才递增
        SendMessage(
            detectDataMsg->rtspDisplayThreadId, MSG_ENCODE_FINISH, nullptr);
        return ACLLITE_OK;
    }
    if (av_log_get_level() != AV_LOG_ERROR)
    {
        av_log_set_level(AV_LOG_ERROR);
    }
    
    for (size_t i = 0; i < detectDataMsg->frame.size(); i++)
    {
        // 数据已在DataOutput中resize,直接使用
        // cv::Mat &frame = detectDataMsg->frame[i];
        // g_picToRtsp.BgrDataToRtsp(frame.data,
        //                           frame.cols * frame.rows * kBgrMultiplier,
        //                           g_frameSeq);  // 使用当前序号
        ImageData imgData = detectDataMsg->decodedImg[i];
        g_picToRtsp.ImageDataToRtsp(imgData, g_frameSeq);  // 使用当前序号
    }
    g_frameSeq++;  // 推流完成后才递增
    return ACLLITE_OK;
}