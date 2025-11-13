#include "pictortsp.h"
#include "AclLiteApp.h"

using namespace cv;
using namespace std;
namespace
{
const string   g_avFormat = "rtsp";
const uint32_t g_bitRate = 1000000;    // 方案3：降低到1Mbps，进一步加快编码速度
const uint32_t g_gopSize = 30;         // 增加GOP，减少关键帧数量，提升性能
const uint16_t g_frameRate = 25;       // 提高到25fps，更接近原始帧率
} // namespace
PicToRtsp::PicToRtsp()
{
    this->g_bgrToRtspFlag = false;
    this->g_yuvToRtspFlag = false;
    this->g_avStream = NULL;
    this->g_codec = NULL;
    this->g_codecCtx = NULL;
    this->g_fmtCtx = NULL;
    this->g_pkt = NULL;
    this->g_imgCtx = NULL;
    this->g_yuvSize = 0;
    this->g_rgbSize = 0;
}

PicToRtsp::~PicToRtsp()
{
    av_packet_free(&g_pkt);
    avcodec_close(g_codecCtx);
    if (g_fmtCtx)
    {
        avio_close(g_fmtCtx->pb);
        avformat_free_context(g_fmtCtx);
    }
}

int PicToRtsp::AvInit(
    int         picWidth,
    int         picHeight,
    std::string g_outFile)
{
    // 设置FFmpeg日志级别，只显示错误信息，隐藏调试信息
    av_log_set_level(AV_LOG_ERROR);
    
    ACLLITE_LOG_INFO("AvInit start: URL=%s, size=%dx%d", 
                     g_outFile.c_str(), picWidth, picHeight);
    
    // 初始化FFmpeg网络组件，用于RTSP等网络协议的支持
    avformat_network_init();
    ACLLITE_LOG_INFO("avformat_network_init completed");

    // 分配输出格式上下文，指定输出格式为RTSP，输出文件路径为g_outFile
    // g_fmtCtx是AVFormatContext指针，用于管理整个媒体文件的格式信息
    ACLLITE_LOG_INFO("Allocating output context for format: %s", g_avFormat.c_str());
    if (avformat_alloc_output_context2(
            &g_fmtCtx,
            NULL,
            g_avFormat.c_str(),
            g_outFile.c_str()) < 0)
    {
        ACLLITE_LOG_ERROR(
            "Cannot alloc output file "
            "context");
        return ACLLITE_ERROR;
    }
    ACLLITE_LOG_INFO("Output context allocated successfully");

    // 设置RTSP传输协议为TCP，确保数据传输的可靠性
    av_opt_set(
        g_fmtCtx->priv_data,
        "rtsp_transport",
        "tcp",
        0);

    // 设置编码器调优参数为"zerolatency"，优先考虑低延迟而不是压缩率
    av_opt_set(
        g_fmtCtx->priv_data,
        "tune",
        "zerolatency",
        0);

    // 设置编码器预设为"superfast"，牺牲一些压缩质量来提高编码速度
    av_opt_set(
        g_fmtCtx->priv_data,
        "preset",
        "superfast",
        0);

    // 查找H264视频编码器，这是最常用的视频编码格式
    g_codec = avcodec_find_encoder(
        AV_CODEC_ID_H264);
    if (g_codec == NULL)
    {
        ACLLITE_LOG_ERROR(
            "Cannot find any endcoder");
        return ACLLITE_ERROR;
    }

    // 为编码器分配上下文结构体，用于存储编码器的配置和状态信息
    g_codecCtx =
        avcodec_alloc_context3(g_codec);
    if (g_codecCtx == NULL)
    {
        ACLLITE_LOG_ERROR(
            "Cannot alloc context");
        return ACLLITE_ERROR;
    }

    // 在输出格式上下文中创建新的视频流
    // 流用于承载特定类型的媒体数据（如视频、音频）
    g_avStream = avformat_new_stream(
        g_fmtCtx,
        g_codec);
    if (g_avStream == NULL)
    {
        ACLLITE_LOG_ERROR(
            "failed create new video "
            "stream");
        return ACLLITE_ERROR;
    }

    // 设置视频流的时间基准为1/g_frameRate，即每帧的时间间隔
    // 时间基准决定了时间戳的单位和精度
    g_avStream->time_base =
        AVRational{1, g_frameRate};

    // 获取当前视频流的编解码器参数结构体
    // codecpar存储了编解码器的基本参数信息
    AVCodecParameters *param =
        g_fmtCtx
            ->streams[g_avStream->index]
            ->codecpar;

    // 设置编解码器参数：媒体类型为视频
    param->codec_type =
        AVMEDIA_TYPE_VIDEO;

    // 设置视频的宽度和高度
    param->width = picWidth;
    param->height = picHeight;

    // 将编解码器参数复制到编码器上下文中
    // 确保编码器上下文与流参数保持一致
    avcodec_parameters_to_context(
        g_codecCtx,
        param);

    // 设置编码器的像素格式为NV12（一种YUV格式，适用于硬件加速）
    g_codecCtx->pix_fmt =
        AV_PIX_FMT_NV12;

    // 设置编码器的时间基准，与视频流的时间基准保持一致
    g_codecCtx->time_base =
        AVRational{1, g_frameRate};

    // 设置视频编码的比特率（码率），影响视频质量和文件大小
    g_codecCtx->bit_rate = g_bitRate;

    // 设置GOP（Group of
    // Pictures）大小，即关键帧之间的帧数
    // GOP越大，压缩率越高，但随机访问性能越差
    g_codecCtx->gop_size = g_gopSize;

    // 设置最大B帧数量为0，简化编码结构，提高实时性
    g_codecCtx->max_b_frames = 0;

    // 如果是H264编码器，设置量化参数范围
    if (g_codecCtx->codec_id ==
        AV_CODEC_ID_H264)
    {
        // 设置最小量化参数（质量最好）
        g_codecCtx->qmin = 10;
        // 设置最大量化参数（质量最差）
        g_codecCtx->qmax = 51;
        // 设置量化压缩系数，影响码率控制
        g_codecCtx->qcompress =
            (float)0.6;
    }

    // 如果是MPEG1视频编码器，设置宏块决策模式
    if (g_codecCtx->codec_id ==
        AV_CODEC_ID_MPEG1VIDEO)
        g_codecCtx->mb_decision = 2;

    // 使用指定的编码器打开编码器上下文，开始编码准备
    ACLLITE_LOG_INFO("Opening codec...");
    if (avcodec_open2(
            g_codecCtx,
            g_codec,
            NULL) < 0)
    {
        ACLLITE_LOG_ERROR(
            "Open encoder failed");
        return ACLLITE_ERROR;
    }
    ACLLITE_LOG_INFO("Codec opened successfully");

    // 将编码器上下文的参数复制回视频流的编解码器参数
    // 确保流参数反映编码器的实际配置
    avcodec_parameters_from_context(
        g_avStream->codecpar,
        g_codecCtx);

    // 输出格式信息到控制台，用于调试和确认配置
    av_dump_format(
        g_fmtCtx,
        0,
        g_outFile.c_str(),
        1);

    ACLLITE_LOG_INFO("Output format flags: 0x%x, AVFMT_NOFILE: %s",
                     g_fmtCtx->oformat->flags,
                     (g_fmtCtx->oformat->flags & AVFMT_NOFILE) ? "YES" : "NO");

    // 对于RTSP等网络协议，需要打开输出流
    // 检查输出格式是否需要IO上下文
    if (!(g_fmtCtx->oformat->flags & AVFMT_NOFILE))
    {
        ACLLITE_LOG_INFO("Opening RTSP output URL: %s", g_outFile.c_str());
        // 打开输出URL
        int ret_open = avio_open2(
            &g_fmtCtx->pb,
            g_outFile.c_str(),
            AVIO_FLAG_WRITE,
            NULL,
            NULL);
        if (ret_open < 0)
        {
            char errbuf[AV_ERROR_MAX_STRING_SIZE];
            av_strerror(ret_open, errbuf, AV_ERROR_MAX_STRING_SIZE);
            ACLLITE_LOG_ERROR(
                "Failed to open output URL: %s, error: %s",
                g_outFile.c_str(), errbuf);
            return ACLLITE_ERROR;
        }
        ACLLITE_LOG_INFO("Successfully opened RTSP output URL");
    }

    // 写入文件头到输出流，初始化RTSP流媒体会话
    // 对于RTSP协议，这会建立网络连接并发送ANNOUNCE等消息
    ACLLITE_LOG_INFO("Writing format header...");
    AVDictionary *opts = NULL;
    int ret = avformat_write_header(
        g_fmtCtx,
        &opts);
    if (ret < 0)
    {
        char errbuf[AV_ERROR_MAX_STRING_SIZE];
        av_strerror(ret, errbuf, AV_ERROR_MAX_STRING_SIZE);
        ACLLITE_LOG_ERROR(
            "Write file header fail, error code: %d, error: %s",
            ret, errbuf);
        if (opts) av_dict_free(&opts);
        return ACLLITE_ERROR;
    }
    if (opts) av_dict_free(&opts);
    ACLLITE_LOG_INFO("Successfully wrote format header, RTSP stream ready");

    // 分配AVPacket结构体，用于存储编码后的数据包
    g_pkt = av_packet_alloc();

    return ACLLITE_OK;
}

int PicToRtsp::FlushEncoder()
{
    int ret;
    int vStreamIndex =
        g_avStream->index;
    AVPacket *pkt = av_packet_alloc();
    pkt->data = NULL;
    pkt->size = 0;

    if (!(g_codecCtx->codec
              ->capabilities &
          AV_CODEC_CAP_DELAY))
    {
        av_packet_free(&pkt);
        return ACLLITE_ERROR;
    }

    ACLLITE_LOG_INFO(
        "Flushing stream %d encoder",
        vStreamIndex);

    if ((ret = avcodec_send_frame(
             g_codecCtx,
             0)) >= 0)
    {
        while (avcodec_receive_packet(
                   g_codecCtx,
                   pkt) >= 0)
        {
            pkt->stream_index =
                vStreamIndex;
            av_packet_rescale_ts(
                pkt,
                g_codecCtx->time_base,
                g_fmtCtx
                    ->streams
                        [vStreamIndex]
                    ->time_base);
            ret =
                av_interleaved_write_frame(
                    g_fmtCtx,
                    pkt);
            if (ret < 0)
            {
                ACLLITE_LOG_ERROR(
                    "error is: %d",
                    ret);
                break;
            }
        }
    }
    av_packet_free(&pkt);
    av_write_trailer(g_fmtCtx);

    if (this->g_bgrToRtspFlag == true)
    {
        av_free(g_brgBuf);
        av_free(g_yuvBuf);
        sws_freeContext(g_imgCtx);
        if (g_rgbFrame)
            av_frame_free(&g_rgbFrame);
        if (g_yuvFrame)
            av_frame_free(&g_yuvFrame);
    }
    if (this->g_yuvToRtspFlag == true)
    {
        av_free(g_yuvBuf);
        if (g_yuvFrame)
            av_frame_free(&g_yuvFrame);
    }
    return ret;
}

void PicToRtsp::YuvDataInit()
{
    if (this->g_yuvToRtspFlag == false)
    {
        g_yuvFrame = av_frame_alloc();
        g_yuvFrame->width =
            g_codecCtx->width;
        g_yuvFrame->height =
            g_codecCtx->height;
        g_yuvFrame->format =
            g_codecCtx->pix_fmt;

        g_yuvSize =
            av_image_get_buffer_size(
                g_codecCtx->pix_fmt,
                g_codecCtx->width,
                g_codecCtx->height,
                1);

        g_yuvBuf = (uint8_t *)av_malloc(
            g_yuvSize);

        int ret = av_image_fill_arrays(
            g_yuvFrame->data,
            g_yuvFrame->linesize,
            g_yuvBuf,
            g_codecCtx->pix_fmt,
            g_codecCtx->width,
            g_codecCtx->height,
            1);
        this->g_yuvToRtspFlag = true;
    }
}

int PicToRtsp::YuvDataToRtsp(
    void    *dataBuf,
    uint32_t size,
    uint32_t seq)
{
    memcpy(g_yuvBuf, dataBuf, size);
    g_yuvFrame->pts = seq;
    if (avcodec_send_frame(
            g_codecCtx,
            g_yuvFrame) >= 0)
    {
        while (avcodec_receive_packet(
                   g_codecCtx,
                   g_pkt) >= 0)
        {
            g_pkt->stream_index =
                g_avStream->index;
            av_packet_rescale_ts(
                g_pkt,
                g_codecCtx->time_base,
                g_avStream->time_base);
            g_pkt->pos = -1;
            int ret =
                av_interleaved_write_frame(
                    g_fmtCtx,
                    g_pkt);
            if (ret < 0)
            {
                char errbuf[AV_ERROR_MAX_STRING_SIZE];
                av_strerror(ret, errbuf, AV_ERROR_MAX_STRING_SIZE);
                ACLLITE_LOG_ERROR(
                    "av_interleaved_write_frame error: %d, %s",
                    ret, errbuf);
            }
        }
    }
    return ACLLITE_OK;
}

void PicToRtsp::BgrDataInint()
{
    if (this->g_bgrToRtspFlag == false)
    {
        g_rgbFrame = av_frame_alloc();
        g_yuvFrame = av_frame_alloc();
        g_rgbFrame->width =
            g_codecCtx->width;
        g_yuvFrame->width =
            g_codecCtx->width;
        g_rgbFrame->height =
            g_codecCtx->height;
        g_yuvFrame->height =
            g_codecCtx->height;
        g_rgbFrame->format =
            AV_PIX_FMT_BGR24;
        g_yuvFrame->format =
            g_codecCtx->pix_fmt;

        g_rgbSize =
            av_image_get_buffer_size(
                AV_PIX_FMT_BGR24,
                g_codecCtx->width,
                g_codecCtx->height,
                1);
        g_yuvSize =
            av_image_get_buffer_size(
                g_codecCtx->pix_fmt,
                g_codecCtx->width,
                g_codecCtx->height,
                1);

        g_brgBuf = (uint8_t *)av_malloc(
            g_rgbSize);
        g_yuvBuf = (uint8_t *)av_malloc(
            g_yuvSize);

        int ret = av_image_fill_arrays(
            g_rgbFrame->data,
            g_rgbFrame->linesize,
            g_brgBuf,
            AV_PIX_FMT_BGR24,
            g_codecCtx->width,
            g_codecCtx->height,
            1);

        ret = av_image_fill_arrays(
            g_yuvFrame->data,
            g_yuvFrame->linesize,
            g_yuvBuf,
            g_codecCtx->pix_fmt,
            g_codecCtx->width,
            g_codecCtx->height,
            1);
        // 使用SWS_FAST_BILINEAR代替SWS_BILINEAR，速度更快
        g_imgCtx = sws_getContext(
            g_codecCtx->width,
            g_codecCtx->height,
            AV_PIX_FMT_BGR24,
            g_codecCtx->width,
            g_codecCtx->height,
            g_codecCtx->pix_fmt,
            SWS_FAST_BILINEAR,  // 更快的算法
            NULL,
            NULL,
            NULL);
        this->g_bgrToRtspFlag = true;
    }
}

int PicToRtsp::BgrDataToRtsp(
    void    *dataBuf,
    uint32_t size,
    uint32_t seq)
{
    static int pushCount = 0;
    static long totalMemcpyTime = 0;
    static long totalSwsScaleTime = 0;
    static long totalEncodeTime = 0;
    static long totalWriteTime = 0;
    pushCount++;
    
    auto startTotal = std::chrono::high_resolution_clock::now();
    
    if (g_rgbSize != size)
    {
        ACLLITE_LOG_ERROR(
            "bgr data size error, The "
            "data size should be %d, "
            "but the actual size is %d",
            g_rgbSize,
            size);
        return ACLLITE_ERROR;
    }
    
    auto start = std::chrono::high_resolution_clock::now();
    memcpy(
        g_brgBuf,
        dataBuf,
        g_rgbSize);
    auto end = std::chrono::high_resolution_clock::now();
    totalMemcpyTime += std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    
    start = std::chrono::high_resolution_clock::now();
    sws_scale(
        g_imgCtx,
        g_rgbFrame->data,
        g_rgbFrame->linesize,
        0,
        g_codecCtx->height,
        g_yuvFrame->data,
        g_yuvFrame->linesize);
    end = std::chrono::high_resolution_clock::now();
    totalSwsScaleTime += std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    
    g_yuvFrame->pts = seq;
    
    start = std::chrono::high_resolution_clock::now();
    if (avcodec_send_frame(
            g_codecCtx,
            g_yuvFrame) >= 0)
    {
        end = std::chrono::high_resolution_clock::now();
        totalEncodeTime += std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
        
        start = std::chrono::high_resolution_clock::now();
        while (avcodec_receive_packet(
                   g_codecCtx,
                   g_pkt) >= 0)
        {
            g_pkt->stream_index =
                g_avStream->index;
            av_packet_rescale_ts(
                g_pkt,
                g_codecCtx->time_base,
                g_avStream->time_base);
            g_pkt->pos = -1;
            int ret =
                av_interleaved_write_frame(
                    g_fmtCtx,
                    g_pkt);
            if (ret < 0)
            {
                ACLLITE_LOG_ERROR(
                    "error is: %d",
                    ret);
            }
        }
        end = std::chrono::high_resolution_clock::now();
        totalWriteTime += std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    }
    
    if (pushCount % 30 == 0) {
        auto endTotal = std::chrono::high_resolution_clock::now();
        auto totalTime = std::chrono::duration_cast<std::chrono::microseconds>(endTotal - startTotal).count();
        ACLLITE_LOG_INFO("[BgrDataToRtsp] Avg times (us): memcpy=%.1f, sws_scale=%.1f, encode=%.1f, write=%.1f, total=%.1f",
                         totalMemcpyTime/30.0, totalSwsScaleTime/30.0, totalEncodeTime/30.0, totalWriteTime/30.0, totalTime/1.0);
        totalMemcpyTime = totalSwsScaleTime = totalEncodeTime = totalWriteTime = 0;
    }
    
    return ACLLITE_OK;
}