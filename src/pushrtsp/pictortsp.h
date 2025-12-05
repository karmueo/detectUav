#pragma once
#include "common.h"
// 注意：必须在common.h之后包含，因为common.h包含opencv，避免命名冲突
#include "../../common/include/VideoWriter.h"
#include <atomic>
#include <condition_variable>
#include <mutex>
#include <queue>
#include <thread>
#include <vector>

// 编码数据包结构
struct H264Packet
{
    std::vector<uint8_t> data;
    uint64_t             pts;
    bool                 isKeyFrame; // 是否为关键帧（I帧）

    H264Packet() : pts(0), isKeyFrame(false) {}
};

class PicToRtsp
{
  public:
    PicToRtsp();
    ~PicToRtsp();

    int AvInit(int picWidth, int picHeight, std::string g_outFile, aclrtContext context = nullptr, VencConfig vencConfig = VencConfig());

    void YuvDataInit();
    void BgrDataInint();

    int YuvDataToRtsp(void *dataBuf, uint32_t size, uint32_t seq);
    int BgrDataToRtsp(void *dataBuf, uint32_t size, uint32_t srcW, uint32_t srcH, uint32_t seq);
    // 直接使用已构造好的 YUV ImageData 进行编码推流（无需尺寸与格式转换）
    int ImageDataToRtsp(ImageData &imageData, uint32_t seq);
    int FlushEncoder();

    // 打印编码相关队列状态（H264输出队列与待编码输入队列）
    void PrintEncodeQueuesStatus();

  private:
    static void VencDataCallbackStatic(void *data, uint32_t size, void *userData);
    void        VencDataCallbackImpl(void *data, uint32_t size);
    void        PushThreadFunc();
    int         PushH264Data(const H264Packet &packet);

    // FFmpeg推流相关（不再用于编码）
    AVFormatContext *g_fmtCtx;
    AVStream        *g_avStream;
    AVPacket        *g_pkt;

    // 硬件编码器（使用全局命名空间的VideoWriter，不是cv::VideoWriter）
    ::VideoWriter *g_videoWriter;
    VencConfig     g_vencConfig;

    // 异步推流队列
    std::queue<H264Packet>  g_h264Queue; // 编码完成后的 H264 数据包队列
    std::mutex              g_queueMutex;
    std::condition_variable g_queueCond;
    std::thread             g_pushThread;
    std::atomic<bool>       g_pushThreadRunning;
    uint64_t                g_frameSeq;
    bool                    g_flushed; // guard repeated flush/free

    // 图像格式转换相关
    AVFrame           *g_rgbFrame;
    uint8_t           *g_brgBuf;
    AVFrame           *g_yuvFrame;
    uint8_t           *g_yuvBuf;
    int                g_yuvSize;
    int                g_rgbSize;
    struct SwsContext *g_imgCtx;
    bool               g_bgrToRtspFlag;
    bool               g_yuvToRtspFlag;
};
