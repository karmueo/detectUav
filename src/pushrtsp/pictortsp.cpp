#include "pictortsp.h"
#include "AclLiteApp.h"
#include <opencv2/imgproc/types_c.h>
#ifdef USE_LIVE555
#include "Live555Streamer.h"
#endif

using namespace cv;
using namespace std;
namespace
{
const string   g_avFormat = "rtsp";
} // namespace
PicToRtsp::PicToRtsp()
{
    this->g_bgrToRtspFlag = false;
    this->g_yuvToRtspFlag = false;
    this->g_avStream = NULL;
    this->g_fmtCtx = NULL;
    this->g_pkt = NULL;
    this->g_imgCtx = NULL;
    this->g_yuvSize = 0;
    this->g_rgbSize = 0;
    this->g_videoWriter = nullptr;
    this->g_frameSeq = 0;
    this->g_pushThreadRunning = false;
    this->g_flushed = false;
}

PicToRtsp::~PicToRtsp()
{
    // 确保释放流程仅执行一次
    FlushEncoder();

    av_packet_free(&g_pkt);
    if (g_fmtCtx)
    {
        if (g_fmtCtx->pb)
        {
            avio_close(g_fmtCtx->pb);
        }
        avformat_free_context(g_fmtCtx);
        g_fmtCtx = nullptr;
    }
    if (g_videoWriter)
    {
        g_videoWriter->Close();
        delete g_videoWriter;
        g_videoWriter = nullptr;
    }
}

int PicToRtsp::AvInit(int picWidth, int picHeight, std::string g_outFile, aclrtContext context, VencConfig vencConfig)
{
    // 设置FFmpeg日志级别
    av_log_set_level(AV_LOG_ERROR);

    ACLLITE_LOG_INFO("AvInit start: URL=%s, size=%dx%d, FPS=%u, GOP=%u, Bitrate=%ukbps, RC=%u",
                     g_outFile.c_str(), picWidth, picHeight, vencConfig.outputFps,
                     vencConfig.gopSize, vencConfig.maxBitrate, vencConfig.rcMode);

    // 1. 配置硬件编码器 - 使用传入的配置并覆盖必要字段
    g_vencConfig = vencConfig;
    g_vencConfig.maxWidth = picWidth;
    g_vencConfig.maxHeight = picHeight;
    g_vencConfig.outFile = "";
    g_vencConfig.format = PIXEL_FORMAT_YUV_SEMIPLANAR_420;
    g_vencConfig.context = context;
    g_vencConfig.dataCallback = VencDataCallbackStatic;
    g_vencConfig.callbackUserData = this;

    g_videoWriter = new ::VideoWriter(g_vencConfig, context);
    AclLiteError ret = g_videoWriter->Open();
    if (ret != ACLLITE_OK)
    {
        ACLLITE_LOG_ERROR("Failed to open hardware encoder");
        delete g_videoWriter;
        g_videoWriter = nullptr;
        return ACLLITE_ERROR;
    }
    ACLLITE_LOG_INFO("Hardware encoder initialized successfully");

    // 2. 初始化FFmpeg网络组件
    avformat_network_init();

    // 3. 分配输出格式上下文
    if (avformat_alloc_output_context2(&g_fmtCtx, NULL, g_avFormat.c_str(), g_outFile.c_str()) < 0)
    {
        ACLLITE_LOG_ERROR("Cannot alloc output file context");
        return ACLLITE_ERROR;
    }

    // 4. 设置RTSP传输优化参数 - 使用配置值
    AVDictionary *format_opts = NULL;
    av_dict_set(&format_opts, "rtsp_transport", g_vencConfig.rtspTransport.c_str(), 0);
    av_dict_set(&format_opts, "buffer_size", std::to_string(g_vencConfig.rtspBufferSize).c_str(), 0);
    av_dict_set(&format_opts, "max_delay", std::to_string(g_vencConfig.rtspMaxDelay).c_str(), 0);
    av_dict_set(&format_opts, "rtsp_flags", g_vencConfig.rtspTransport == "tcp" ? "prefer_tcp" : "prefer_udp", 0);
    
    // 将参数应用到私有数据
    if (g_fmtCtx->priv_data) {
        av_opt_set(g_fmtCtx->priv_data, "rtsp_transport", g_vencConfig.rtspTransport.c_str(), 0);
        av_opt_set(g_fmtCtx->priv_data, "buffer_size", std::to_string(g_vencConfig.rtspBufferSize).c_str(), 0);
    }

    // 5. 创建H264视频流
    g_avStream = avformat_new_stream(g_fmtCtx, NULL);
    if (g_avStream == NULL)
    {
        ACLLITE_LOG_ERROR("failed create new video stream");
        av_dict_free(&format_opts);
        return ACLLITE_ERROR;
    }

    // 6. 设置视频流时间基准 - 使用更高精度的时间基准和配置的帧率
    g_avStream->time_base = AVRational{1, 90000};  // H.264标准时间基准
    g_avStream->avg_frame_rate = AVRational{(int)g_vencConfig.outputFps, 1};
    g_avStream->r_frame_rate = AVRational{(int)g_vencConfig.outputFps, 1};

    // 7. 配置编解码器参数
    AVCodecParameters *param = g_avStream->codecpar;
    param->codec_type = AVMEDIA_TYPE_VIDEO;
    param->codec_id = AV_CODEC_ID_H264;
    param->codec_tag = 0;  // 让FFmpeg自动选择
    param->width = picWidth;
    param->height = picHeight;
    param->format = AV_PIX_FMT_YUV420P;
    param->bit_rate = g_vencConfig.maxBitrate * 1000;  // kbps转bps
    
    // 8. 输出格式信息
    av_dump_format(g_fmtCtx, 0, g_outFile.c_str(), 1);

    // 9. 打开RTSP输出流
    if (!(g_fmtCtx->oformat->flags & AVFMT_NOFILE))
    {
        ACLLITE_LOG_INFO("Opening RTSP output URL: %s", g_outFile.c_str());
        int ret_open = avio_open2(&g_fmtCtx->pb, g_outFile.c_str(), AVIO_FLAG_WRITE, NULL, &format_opts);
        if (ret_open < 0)
        {
            char errbuf[AV_ERROR_MAX_STRING_SIZE];
            av_strerror(ret_open, errbuf, AV_ERROR_MAX_STRING_SIZE);
            ACLLITE_LOG_ERROR("Failed to open output URL: %s, error: %s", g_outFile.c_str(), errbuf);
            av_dict_free(&format_opts);
            return ACLLITE_ERROR;
        }
        ACLLITE_LOG_INFO("Successfully opened RTSP output URL");
    }
    av_dict_free(&format_opts);

    // 10. 写入RTSP流头
    ACLLITE_LOG_INFO("Writing format header...");
    AVDictionary *header_opts = NULL;
    av_dict_set(&header_opts, "rtsp_flags", "prefer_tcp", 0);
    
    int ret_header = avformat_write_header(g_fmtCtx, &header_opts);
    if (ret_header < 0)
    {
        char errbuf[AV_ERROR_MAX_STRING_SIZE];
        av_strerror(ret_header, errbuf, AV_ERROR_MAX_STRING_SIZE);
        ACLLITE_LOG_ERROR("Write file header fail, error code: %d, error: %s", ret_header, errbuf);
        av_dict_free(&header_opts);
        return ACLLITE_ERROR;
    }
    av_dict_free(&header_opts);
    ACLLITE_LOG_INFO("Successfully wrote format header, RTSP stream ready");

    // 11. 分配AVPacket
    g_pkt = av_packet_alloc();

    // 12. 启动异步推流线程
    g_pushThreadRunning = true;
    g_pushThread = std::thread(&PicToRtsp::PushThreadFunc, this);

    ACLLITE_LOG_INFO("RTSP stream initialization completed");
    return ACLLITE_OK;
}

// 静态回调函数，由VencHelper.cpp中的DvppVenc::SaveVencFile调用
// NOTE: data指向编码后的H264数据
void PicToRtsp::VencDataCallbackStatic(void* data, uint32_t size, void* userData)
{
    PicToRtsp* instance = static_cast<PicToRtsp*>(userData);
    if (instance)
    {
        instance->VencDataCallbackImpl(data, size);
    }
}

// 实例回调函数，处理编码数据
void PicToRtsp::VencDataCallbackImpl(void* data, uint32_t size)
{
    if (data == nullptr || size == 0)
    {
        return;
    }
    
    // 将编码数据放入队列，避免阻塞编码线程
    H264Packet packet;
    packet.data.resize(size);
    memcpy(packet.data.data(), data, size);
    packet.pts = g_frameSeq++;
    
    // 检测是否为关键帧（I帧）：H264 NAL type 5 = IDR slice
    packet.isKeyFrame = false;
    if (size >= 5) {
        // 查找 NAL start code (0x00 0x00 0x00 0x01 或 0x00 0x00 0x01)
        uint8_t* buf = static_cast<uint8_t*>(data);
        size_t nalStart = 0;
        if (buf[0] == 0 && buf[1] == 0) {
            if (buf[2] == 1) nalStart = 3;
            else if (buf[2] == 0 && buf[3] == 1) nalStart = 4;
        }
        if (nalStart > 0 && nalStart < size) {
            uint8_t nalType = buf[nalStart] & 0x1F;
            packet.isKeyFrame = (nalType == 5 || nalType == 7 || nalType == 8); // IDR, SPS, PPS
            if (packet.isKeyFrame) {
                static int keyFrameCount = 0;
                if (++keyFrameCount % 50 == 1) {
                    ACLLITE_LOG_INFO("Key frame #%d detected, NAL type: %d, size: %u bytes", keyFrameCount, nalType, size);
                }
            }
        }
    }
    
    {
        std::lock_guard<std::mutex> lock(g_queueMutex);
        g_h264Queue.push(std::move(packet));
        
        // 限制队列大小，防止内存无限增长（适度增大，减少丢帧带来的画面抖动，代价是更高的缓冲/时延）
        const size_t MAX_QUEUE_SIZE = 150;
        while (g_h264Queue.size() > MAX_QUEUE_SIZE)
        {
            static size_t dropCnt = 0;
            if (++dropCnt % 50 == 1) {
                ACLLITE_LOG_WARNING("H264 queue overflow: size=%zu>%zu, dropped %zu frames",
                                    g_h264Queue.size(), MAX_QUEUE_SIZE, dropCnt);
            }
            g_h264Queue.pop();
        }
    }
    g_queueCond.notify_one();
}

// 异步推流线程
void PicToRtsp::PushThreadFunc()
{
    ACLLITE_LOG_INFO("Push thread started");
    
    while (g_pushThreadRunning)
    {
        H264Packet packet;
        {
            std::unique_lock<std::mutex> lock(g_queueMutex);
            g_queueCond.wait(lock, [this] { 
                return !g_h264Queue.empty() || !g_pushThreadRunning; 
            });
            
            if (!g_pushThreadRunning && g_h264Queue.empty())
            {
                break;
            }
            
            if (g_h264Queue.empty())
            {
                continue;
            }
            
            packet = std::move(g_h264Queue.front());
            g_h264Queue.pop();
        }
        
        // 在锁外执行推流，避免阻塞队列
        PushH264Data(packet);
    }
    
    ACLLITE_LOG_INFO("Push thread exited");
}

// 推送H264数据到RTSP流
int PicToRtsp::PushH264Data(const H264Packet& packet)
{
#ifdef USE_LIVE555
    // 使用 Live555: 在此直接将数据投喂给 Live555 内置队列
    static Live555Streamer s_streamer;
    static bool s_inited = false;
    if (!s_inited) {
        // 首次调用时初始化本地 RTSP 服务器(默认 8554/stream)，使用配置的帧率
        if (!s_streamer.InitStandalone(8554, "stream", g_vencConfig.outputFps)) {
            ACLLITE_LOG_ERROR("Live555 InitStandalone failed");
            return ACLLITE_ERROR;
        }
        s_streamer.Start();
        ACLLITE_LOG_INFO("Live555 started at %s", s_streamer.GetRtspUrl().c_str());
        s_inited = true;
    }

    s_streamer.Enqueue(packet);
    return ACLLITE_OK;
#else
    if (!g_fmtCtx || !g_avStream || !g_pkt)
    {
        ACLLITE_LOG_ERROR("Invalid RTSP context");
        return ACLLITE_ERROR;
    }

    // 封装H264数据到AVPacket
    av_packet_unref(g_pkt);
    
    // 分配新的缓冲区并复制数据
    g_pkt->data = (uint8_t*)av_malloc(packet.data.size());
    if (!g_pkt->data)
    {
        ACLLITE_LOG_ERROR("Failed to allocate packet data");
        return ACLLITE_ERROR;
    }
    
    memcpy(g_pkt->data, packet.data.data(), packet.data.size());
    g_pkt->size = packet.data.size();
    g_pkt->stream_index = g_avStream->index;
    
    // 正确计算时间戳：将帧序号转换为90000Hz时间基准
    int64_t pts = packet.pts * (90000 / g_frameRate);
    g_pkt->pts = pts;
    g_pkt->dts = pts;
    g_pkt->duration = 90000 / g_frameRate; // 每帧的持续时间
    g_pkt->pos = -1;
    
    // 检测并标记关键帧(I帧)
    if (packet.data.size() >= 5)
    {
        size_t nalStart = 0;
        if (packet.data[0] == 0 && packet.data[1] == 0)
        {
            if (packet.data[2] == 1) nalStart = 3;
            else if (packet.data[2] == 0 && packet.data[3] == 1) nalStart = 4;
        }
        if (nalStart > 0 && nalStart < packet.data.size())
        {
            uint8_t nalType = packet.data[nalStart] & 0x1F;
            if (nalType == 5 || nalType == 7 || nalType == 8) g_pkt->flags |= AV_PKT_FLAG_KEY;
        }
    }

    int ret = av_interleaved_write_frame(g_fmtCtx, g_pkt);
    if (ret < 0)
    {
        char errbuf[AV_ERROR_MAX_STRING_SIZE];
        av_strerror(ret, errbuf, AV_ERROR_MAX_STRING_SIZE);
        ACLLITE_LOG_ERROR("av_interleaved_write_frame error: %d, %s", ret, errbuf);
        av_freep(&g_pkt->data);
        return ACLLITE_ERROR;
    }
    av_freep(&g_pkt->data);
    return ACLLITE_OK;
#endif
}

int PicToRtsp::FlushEncoder()
{
    ACLLITE_LOG_INFO("Flushing encoder and closing RTSP stream");

    // 避免重复执行资源释放
    if (g_flushed)
    {
        return ACLLITE_OK;
    }

    // 先停推流线程，防止线程在释放 FFmpeg 资源后仍访问
    if (g_pushThreadRunning)
    {
        g_pushThreadRunning = false;
        g_queueCond.notify_all();
        if (g_pushThread.joinable())
        {
            g_pushThread.join();
            ACLLITE_LOG_INFO("Push thread exited");
        }
    }

    // 写入RTSP流尾
    if (g_fmtCtx)
    {
        av_write_trailer(g_fmtCtx);
    }

    // 清理资源
    if (this->g_bgrToRtspFlag == true)
    {
        if (g_brgBuf)
        {
            av_free(g_brgBuf);
            g_brgBuf = nullptr;
        }
        if (g_yuvBuf)
        {
            av_free(g_yuvBuf);
            g_yuvBuf = nullptr;
        }
        if (g_imgCtx)
        {
            sws_freeContext(g_imgCtx);
            g_imgCtx = nullptr;
        }
        if (g_rgbFrame)
        {
            av_frame_free(&g_rgbFrame);
        }
        if (g_yuvFrame)
        {
            av_frame_free(&g_yuvFrame);
        }
        this->g_bgrToRtspFlag = false;
    }
    else if (this->g_yuvToRtspFlag == true)
    {
        if (g_yuvBuf)
        {
            av_free(g_yuvBuf);
            g_yuvBuf = nullptr;
        }
        this->g_yuvToRtspFlag = false;
    }

    g_flushed = true;
    return ACLLITE_OK;
}

void PicToRtsp::YuvDataInit()
{
    if (this->g_yuvToRtspFlag == false)
    {
        g_yuvSize = g_vencConfig.maxWidth * g_vencConfig.maxHeight * 3 / 2;
        g_yuvBuf = (uint8_t *)av_malloc(g_yuvSize);
        this->g_yuvToRtspFlag = true;
    }
}

int PicToRtsp::YuvDataToRtsp(void *dataBuf, uint32_t size, uint32_t seq)
{
    // 使用硬件编码器编码YUV数据
    ImageData imageData;
    imageData.format = PIXEL_FORMAT_YUV_SEMIPLANAR_420;
    imageData.width = g_vencConfig.maxWidth;
    imageData.height = g_vencConfig.maxHeight;
    imageData.size = size;
    // 直接使用传入的dataBuf，不拷贝
    imageData.data = std::shared_ptr<uint8_t>((uint8_t *)dataBuf, [](uint8_t *) {});

    AclLiteError ret = g_videoWriter->Read(imageData);
    if (ret != ACLLITE_OK)
    {
        ACLLITE_LOG_ERROR("Hardware encode YUV failed");
        return ACLLITE_ERROR;
    }

    // 编码完成后，回调函数会自动被调用并推流
    return ACLLITE_OK;
}

// 直接使用传入的 YUV ImageData 进行编码推流（无需尺寸与格式转换）
int PicToRtsp::ImageDataToRtsp(ImageData& imageData, uint32_t seq)
{
    /* {
        cv::Mat frame;
        cv::Mat yuvimg(imageData.height * 3 / 2, imageData.width, CV_8UC1, imageData.data.get());
        cv::cvtColor(yuvimg, frame, CV_YUV2BGR_NV12);
        // 保存
        cv::imwrite("frame_tmp.jpg", frame);
    } */
    static int  pushCount = 0;
    static long totalEncodeTime = 0;
    pushCount++;

    auto start = std::chrono::high_resolution_clock::now();

    if (g_videoWriter == nullptr)
    {
        ACLLITE_LOG_ERROR("Hardware encoder not initialized");
        return ACLLITE_ERROR;
    }

    // 如果 ImageData 大小与编码器期望大小不符,尝试做 resize (fallback)
    if (imageData.width != g_vencConfig.maxWidth || imageData.height != g_vencConfig.maxHeight) {
        ACLLITE_LOG_INFO("ImageData size (%u x %u) != encoder size (%u x %u), apply fallback sws resize",
                         imageData.width, imageData.height, g_vencConfig.maxWidth, g_vencConfig.maxHeight);
        int srcW = imageData.width;
        int srcH = imageData.height;
        int dstW = g_vencConfig.maxWidth;
        int dstH = g_vencConfig.maxHeight;
        int dstSize = av_image_get_buffer_size(AV_PIX_FMT_NV12, dstW, dstH, 1);
        uint8_t *dstBuf = (uint8_t *)av_malloc(dstSize);
        if (dstBuf == nullptr) {
            ACLLITE_LOG_ERROR("Unable to allocate buffer for YUV resize fallback");
            return ACLLITE_ERROR;
        }
        AVFrame *srcFrame = av_frame_alloc();
        AVFrame *dstFrame = av_frame_alloc();
        av_image_fill_arrays(srcFrame->data, srcFrame->linesize, imageData.data.get(), AV_PIX_FMT_NV12, srcW, srcH, 1);
        av_image_fill_arrays(dstFrame->data, dstFrame->linesize, dstBuf, AV_PIX_FMT_NV12, dstW, dstH, 1);
        SwsContext *sws = sws_getContext(srcW, srcH, AV_PIX_FMT_NV12, dstW, dstH, AV_PIX_FMT_NV12, SWS_FAST_BILINEAR, NULL, NULL, NULL);
        if (!sws) {
            av_free(dstBuf);
            av_frame_free(&srcFrame);
            av_frame_free(&dstFrame);
            ACLLITE_LOG_ERROR("Failed to create sws context for YUV fallback resize");
            return ACLLITE_ERROR;
        }
        sws_scale(sws, srcFrame->data, srcFrame->linesize, 0, srcH, dstFrame->data, dstFrame->linesize);
        sws_freeContext(sws);
        av_frame_free(&srcFrame);
        av_frame_free(&dstFrame);
        ImageData tmp;
        tmp.format = PIXEL_FORMAT_YUV_SEMIPLANAR_420;
        tmp.width = dstW;
        tmp.height = dstH;
        tmp.size = dstSize;
        tmp.data = std::shared_ptr<uint8_t>(dstBuf, [](uint8_t *p) { av_free(p); });
        AclLiteError ret = g_videoWriter->Read(tmp);
        if (ret != ACLLITE_OK) {
            ACLLITE_LOG_ERROR("Hardware encode YUV(ImageData) failed after fallback resize");
            return ACLLITE_ERROR;
        }
    } else {
        AclLiteError ret = g_videoWriter->Read(imageData);
        if (ret != ACLLITE_OK) {
            ACLLITE_LOG_ERROR("Hardware encode YUV(ImageData) failed");
            return ACLLITE_ERROR;
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    totalEncodeTime += std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

    // 周期性打印队列状态，保持与 BgrDataToRtsp 一致的可观测性
    if (pushCount % 30 == 0)
    {
        size_t queueSize = 0;
        {
            std::lock_guard<std::mutex> lock(g_queueMutex);
            queueSize = g_h264Queue.size();
        }
        uint32_t vencInputQueueSize = (g_videoWriter != nullptr) ? g_videoWriter->GetInputQueueSize() : 0;

        ACLLITE_LOG_INFO(
            "[ImageDataToRtsp] Avg (us): hw_encode=%.1f, h264Queue=%zu, vencInputQueue=%u",
            totalEncodeTime / 30.0,
            queueSize,
            vencInputQueueSize);
        totalEncodeTime = 0;
    }

    // 编码完成后，回调函数会自动被调用并推流
    return ACLLITE_OK;
}

void PicToRtsp::BgrDataInint()
{
    if (this->g_bgrToRtspFlag == false)
    {
        g_rgbFrame = av_frame_alloc();
        g_yuvFrame = av_frame_alloc();
        g_rgbFrame->width = g_vencConfig.maxWidth;
        g_yuvFrame->width = g_vencConfig.maxWidth;
        g_rgbFrame->height = g_vencConfig.maxHeight;
        g_yuvFrame->height = g_vencConfig.maxHeight;
        g_rgbFrame->format = AV_PIX_FMT_BGR24;
        g_yuvFrame->format = AV_PIX_FMT_NV12;

        g_rgbSize = av_image_get_buffer_size(AV_PIX_FMT_BGR24, g_vencConfig.maxWidth, g_vencConfig.maxHeight, 1);
        g_yuvSize = av_image_get_buffer_size(AV_PIX_FMT_NV12, g_vencConfig.maxWidth, g_vencConfig.maxHeight, 1);

        g_brgBuf = (uint8_t *)av_malloc(g_rgbSize);
        g_yuvBuf = (uint8_t *)av_malloc(g_yuvSize);

        av_image_fill_arrays(
            g_rgbFrame->data,
            g_rgbFrame->linesize,
            g_brgBuf,
            AV_PIX_FMT_BGR24,
            g_vencConfig.maxWidth,
            g_vencConfig.maxHeight,
            1);

        av_image_fill_arrays(
            g_yuvFrame->data,
            g_yuvFrame->linesize,
            g_yuvBuf,
            AV_PIX_FMT_NV12,
            g_vencConfig.maxWidth,
            g_vencConfig.maxHeight,
            1);

        // BGR转YUV的转换器
        g_imgCtx = sws_getContext(
            g_vencConfig.maxWidth,
            g_vencConfig.maxHeight,
            AV_PIX_FMT_BGR24,
            g_vencConfig.maxWidth,
            g_vencConfig.maxHeight,
            AV_PIX_FMT_NV12,
            SWS_FAST_BILINEAR,
            NULL,
            NULL,
            NULL);
        this->g_bgrToRtspFlag = true;
    }
}

// FIXME: 速度瓶颈
int PicToRtsp::BgrDataToRtsp(void *dataBuf, uint32_t size, uint32_t srcW, uint32_t srcH, uint32_t seq)
{
    static int  pushCount = 0;
    static long totalConvertTime = 0;
    static long totalEncodeTime = 0;
    pushCount++;

    auto startTotal = std::chrono::high_resolution_clock::now();

    uint32_t expected_rgb_size = srcW * srcH * 3;
    if (expected_rgb_size != size)
    {
        ACLLITE_LOG_ERROR(
            "bgr data size error, The "
            "data size should be %d, "
            "but the actual size is %d",
            g_rgbSize,
            size);
        return ACLLITE_ERROR;
    }

    // 1. BGR转YUV (使用FFmpeg的sws_scale)
    auto start = std::chrono::high_resolution_clock::now();
    // create src frame referencing dataBuf
    AVFrame *srcFrame = av_frame_alloc();
    av_image_fill_arrays(srcFrame->data, srcFrame->linesize, (uint8_t *)dataBuf, AV_PIX_FMT_BGR24, srcW, srcH, 1);

    // ensure g_imgCtx configured for current src->dst sizes
    SwsContext *localCtx = sws_getContext(srcW, srcH, AV_PIX_FMT_BGR24, g_vencConfig.maxWidth, g_vencConfig.maxHeight, AV_PIX_FMT_NV12, SWS_FAST_BILINEAR, NULL, NULL, NULL);
    if (!localCtx) {
        av_frame_free(&srcFrame);
        ACLLITE_LOG_ERROR("Failed to create sws context for BGR->NV12");
        return ACLLITE_ERROR;
    }
    sws_scale(
        localCtx,
        srcFrame->data,
        srcFrame->linesize,
        0,
        srcH,
        g_yuvFrame->data,
        g_yuvFrame->linesize);
    sws_freeContext(localCtx);
    av_frame_free(&srcFrame);
    auto end = std::chrono::high_resolution_clock::now();
    totalConvertTime += std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

    // 2. 使用硬件编码器编码YUV数据
    start = std::chrono::high_resolution_clock::now();
    ImageData imageData;
    imageData.format = PIXEL_FORMAT_YUV_SEMIPLANAR_420;
    imageData.width = g_vencConfig.maxWidth;
    imageData.height = g_vencConfig.maxHeight;
    imageData.size = g_yuvSize;
    imageData.data = std::shared_ptr<uint8_t>(g_yuvBuf, [](uint8_t *) {});

    AclLiteError ret = g_videoWriter->Read(imageData);
    if (ret != ACLLITE_OK)
    {
        ACLLITE_LOG_ERROR("Hardware encode failed");
        return ACLLITE_ERROR;
    }
    end = std::chrono::high_resolution_clock::now();
    totalEncodeTime += std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

    // 3. 编码完成后，回调函数会自动被调用并将数据放入队列

    if (pushCount % 30 == 0)
    {
        size_t queueSize = 0;
        {
            std::lock_guard<std::mutex> lock(g_queueMutex);
            queueSize = g_h264Queue.size();
        }
        // 获取待编码输入队列大小
        uint32_t vencInputQueueSize = (g_videoWriter != nullptr) ? g_videoWriter->GetInputQueueSize() : 0;

        auto endTotal = std::chrono::high_resolution_clock::now();
        auto totalTime = std::chrono::duration_cast<std::chrono::microseconds>(endTotal - startTotal).count();
        ACLLITE_LOG_INFO(
            "[BgrDataToRtsp] Avg (us): bgr2yuv=%.1f, hw_encode=%.1f, total=%.1f, h264Queue=%zu, vencInputQueue=%u",
            totalConvertTime / 30.0,
            totalEncodeTime / 30.0,
            totalTime / 1.0,
            queueSize,
            vencInputQueueSize);
        totalConvertTime = totalEncodeTime = 0;
    }

    return ACLLITE_OK;
}

void PicToRtsp::PrintEncodeQueuesStatus()
{
    size_t h264Size = 0;
    {
        std::lock_guard<std::mutex> lock(g_queueMutex);
        h264Size = g_h264Queue.size();
    }
    uint32_t vencInputQueueSize = (g_videoWriter != nullptr) ? g_videoWriter->GetInputQueueSize() : 0;
    ACLLITE_LOG_INFO("========== Encode Queue Status ==========");
    ACLLITE_LOG_INFO("  [PicToRtsp] H264 output queue: %zu", h264Size);
    ACLLITE_LOG_INFO("  [VencHelper] Input frame queue: %u", vencInputQueueSize);
    ACLLITE_LOG_INFO("=========================================");
}