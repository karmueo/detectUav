#include "pushrtspthread.h"
#include "AclLiteApp.h"
#include <chrono>
#include <cstddef>
using namespace cv;
using namespace std;
namespace
{
uint32_t kBgrMultiplier = 3;
} // namespace

PushRtspThread::PushRtspThread(std::string rtspUrl, VencConfig vencConfig)
{
    g_rtspUrl = rtspUrl;
    g_vencConfig = vencConfig;
    ACLLITE_LOG_INFO("PushRtspThread URL: %s, Resolution: %ux%u, FPS: %u, GOP: %u, Bitrate: %u kbps, RC Mode: %u",
                     g_rtspUrl.c_str(), g_vencConfig.outputWidth, g_vencConfig.outputHeight,
                     g_vencConfig.outputFps, g_vencConfig.gopSize, g_vencConfig.maxBitrate, g_vencConfig.rcMode);
}

// FlushEncoder will be called in PicToRtsp destructor
PushRtspThread::~PushRtspThread() {}

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
    
    AclLiteError ret = g_picToRtsp.AvInit(g_vencConfig.outputWidth, g_vencConfig.outputHeight, g_rtspUrl, context, g_vencConfig);
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
                                      frame.cols,
                                      frame.rows,
                                      g_frameSeq++);
        }
        SendMessage(
            detectDataMsg->rtspDisplayThreadId, MSG_ENCODE_FINISH, nullptr);
        return ACLLITE_OK;
    }
    if (av_log_get_level() != AV_LOG_ERROR)
    {
        av_log_set_level(AV_LOG_ERROR);
    }

    // NOTE: 发送YUV数据进行编码推流
    for (size_t i = 0; i < detectDataMsg->frame.size(); i++)
    {
        // 数据已在DataOutput中resize,直接使用
        // cv::Mat &frame = detectDataMsg->frame[i];
        // g_picToRtsp.BgrDataToRtsp(frame.data,
        //                           frame.cols * frame.rows * kBgrMultiplier,
        //                           g_frameSeq++);
        ImageData imgData = detectDataMsg->decodedImg[i];
        g_picToRtsp.ImageDataToRtsp(imgData, g_frameSeq++);
    }
    return ACLLITE_OK;
}