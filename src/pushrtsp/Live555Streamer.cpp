#include "Live555Streamer.h"
#include "AclLiteApp.h"

// 注意: 只在 .cpp 文件中包含 live555 头文件
#ifdef USE_LIVE555
#include <BasicUsageEnvironment.hh>
#include <GroupsockHelper.hh>
#include <H264VideoRTPSink.hh>
#include <H264VideoStreamFramer.hh>
#include <OnDemandServerMediaSubsession.hh>
#include <liveMedia.hh>

#include <arpa/inet.h>
#include <cstring>
#include <netinet/in.h>
#include <sys/socket.h>
#include <thread>
#include <unistd.h>

namespace
{
std::string GetLocalHostIp()
{
    static std::string ip;
    if (!ip.empty())
        return ip;

    // 优先尝试通过外网连接探测 IP
    int sock = socket(AF_INET, SOCK_DGRAM, 0);
    if (sock >= 0)
    {
        struct sockaddr_in remote{};
        remote.sin_family = AF_INET;
        remote.sin_port = htons(80);
        inet_pton(AF_INET, "8.8.8.8", &remote.sin_addr);

        if (connect(sock, reinterpret_cast<struct sockaddr *>(&remote), sizeof(remote)) == 0)
        {
            struct sockaddr_in local{};
            socklen_t len = sizeof(local);
            if (getsockname(sock, reinterpret_cast<struct sockaddr *>(&local), &len) == 0)
            {
                char buffer[INET_ADDRSTRLEN] = {0};
                inet_ntop(AF_INET, &local.sin_addr, buffer, sizeof(buffer));
                if (buffer[0] != '\0')
                {
                    std::string candidate = buffer;
                    // 过滤 Docker/环回地址
                    if (candidate.find("127.") != 0 && candidate.find("172.17.") != 0)
                    {
                        ip = candidate;
                    }
                }
            }
        }
        close(sock);
    }

    // 回退到 localhost
    if (ip.empty())
        ip = "localhost";

    return ip;
}
} // namespace

// ==================== Live555H264Source 实现 ====================

class Live555H264Source : public FramedSource
{
  public:
    static Live555H264Source *createNew(
        UsageEnvironment        &env,
        std::queue<H264Packet>  *queue,
        std::mutex              *queueMutex,
        std::condition_variable *queueCond,
        std::atomic<bool>       *running,
        unsigned                 fps);

  protected:
    Live555H264Source(
        UsageEnvironment        &env,
        std::queue<H264Packet>  *queue,
        std::mutex              *queueMutex,
        std::condition_variable *queueCond,
        std::atomic<bool>       *running,
        unsigned                 fps);

    virtual ~Live555H264Source();

  private:
    virtual void doGetNextFrame() override;
    static void  deliverFrame0(void *clientData);
    void         deliverFrame();

    std::queue<H264Packet>  *fQueue;
    std::mutex              *fQueueMutex;
    std::condition_variable *fQueueCond;
    std::atomic<bool>       *fRunning;

    unsigned       fFrameDuration; // 微秒
    struct timeval fLastFrameTime;
    bool           fHaveStartedReading;
    
    // 帧分片支持：处理大关键帧
    H264Packet     fCurrentPacket;
    size_t         fPacketOffset;  // 当前包的发送偏移
    bool           fHasPartialFrame;
};

Live555H264Source *Live555H264Source::createNew(
    UsageEnvironment        &env,
    std::queue<H264Packet>  *queue,
    std::mutex              *queueMutex,
    std::condition_variable *queueCond,
    std::atomic<bool>       *running,
    unsigned                 fps)
{
    return new Live555H264Source(env, queue, queueMutex, queueCond, running, fps);
}

Live555H264Source::Live555H264Source(
    UsageEnvironment        &env,
    std::queue<H264Packet>  *queue,
    std::mutex              *queueMutex,
    std::condition_variable *queueCond,
    std::atomic<bool>       *running,
    unsigned                 fps)
    : FramedSource(env),
      fQueue(queue),
      fQueueMutex(queueMutex),
      fQueueCond(queueCond),
      fRunning(running),
      fFrameDuration(1000000 / fps),
      fHaveStartedReading(false),
      fPacketOffset(0),
      fHasPartialFrame(false)
{
    gettimeofday(&fLastFrameTime, NULL);
    ACLLITE_LOG_INFO("Live555H264Source created, fps=%u, frameDuration=%uus", fps, fFrameDuration);
}

Live555H264Source::~Live555H264Source() { ACLLITE_LOG_INFO("Live555H264Source destroyed"); }

void Live555H264Source::doGetNextFrame()
{
    if (!fHaveStartedReading)
    {
        fHaveStartedReading = true;
    }

    deliverFrame();
}

void Live555H264Source::deliverFrame0(void *clientData) { ((Live555H264Source *)clientData)->deliverFrame(); }

void Live555H264Source::deliverFrame()
{
    if (!isCurrentlyAwaitingData())
    {
        return;
    }

    // 如果有未完成的帧，继续发送
    if (!fHasPartialFrame)
    {
        // 尝试从队列获取新数据
        std::unique_lock<std::mutex> lock(*fQueueMutex, std::try_to_lock);
        if (lock.owns_lock() && !fQueue->empty())
        {
            fCurrentPacket = std::move(fQueue->front());
            fQueue->pop();
            fPacketOffset = 0;
            fHasPartialFrame = true;
            
            static int frameCount = 0;
            if (++frameCount % 100 == 0)
            {
                ACLLITE_LOG_INFO("Delivered %d frames, queue size: %zu", frameCount, fQueue->size());
            }
        }
        else
        {
            // 队列为空,延迟重试
            int64_t uSecsToDelay = 10000; // 10ms
            nextTask() = envir().taskScheduler().scheduleDelayedTask(uSecsToDelay, (TaskFunc *)deliverFrame0, this);
            return;
        }
    }

    // 计算本次可以发送的数据大小
    size_t remainingSize = fCurrentPacket.data.size() - fPacketOffset;
    size_t sendSize = (remainingSize > fMaxSize) ? fMaxSize : remainingSize;
    
    // 复制数据
    memcpy(fTo, fCurrentPacket.data.data() + fPacketOffset, sendSize);
    fFrameSize = sendSize;
    fNumTruncatedBytes = 0;
    
    fPacketOffset += sendSize;
    
    // 检查是否发送完成
    if (fPacketOffset >= fCurrentPacket.data.size())
    {
        fHasPartialFrame = false;
        fPacketOffset = 0;
        
        // 如果是大帧且已完成，记录日志
        if (fCurrentPacket.data.size() > fMaxSize && fCurrentPacket.isKeyFrame)
        {
            static int fragmentCount = 0;
            if (++fragmentCount % 10 == 1)
            {
                ACLLITE_LOG_INFO("Large key frame sent in fragments: %zu bytes in %zu chunks",
                    fCurrentPacket.data.size(), (fCurrentPacket.data.size() + fMaxSize - 1) / fMaxSize);
            }
        }
    }

    // 计算时间戳(使用固定帧间隔保证平滑播放)
    struct timeval now;
    gettimeofday(&now, NULL);

    unsigned timeSinceLastFrame =
        (now.tv_sec - fLastFrameTime.tv_sec) * 1000000 + (now.tv_usec - fLastFrameTime.tv_usec);

    if (timeSinceLastFrame < fFrameDuration)
    {
        // 帧到达太快,延迟投递保持恒定帧率
        int64_t uSecsToDelay = fFrameDuration - timeSinceLastFrame;

        // 数据已复制到 fTo,延迟后直接通知
        nextTask() = envir().taskScheduler().scheduleDelayedTask(
            uSecsToDelay,
            [](void *clientData)
            {
                Live555H264Source *source = (Live555H264Source *)clientData;
                source->fLastFrameTime = source->fPresentationTime;
                gettimeofday(&source->fPresentationTime, NULL);
                source->fDurationInMicroseconds = source->fFrameDuration;
                FramedSource::afterGetting(source);
            },
            this);
        return;
    }

    fLastFrameTime = now;
    fPresentationTime = now;
    fDurationInMicroseconds = fFrameDuration;

    // 通知 live555 数据已就绪
    afterGetting(this);
}

// ==================== Live555Streamer 实现 ====================

Live555Streamer::Live555Streamer()
    : fScheduler(nullptr),
      fEnv(nullptr),
      fRtspServer(nullptr),
      fSms(nullptr),
      fRtspPort(8554),
      fStreamName("stream"),
      fFps(25),
      fH264Queue(nullptr),
      fQueueMutex(nullptr),
      fQueueCond(nullptr),
      fRunning(nullptr),
      fInternalRunning(false),
      fUseInternalQueue(false),
      fEventLoopRunning(false),
      fInitialized(false)
{
    fEventLoopStopFlag.store(0);
}

Live555Streamer::~Live555Streamer() { Stop(); }

bool Live555Streamer::Init(
    std::queue<H264Packet>  *queue,
    std::mutex              *queueMutex,
    std::condition_variable *queueCond,
    std::atomic<bool>       *running,
    int                      rtspPort,
    const std::string       &streamName,
    unsigned                 fps)
{
    if (fInitialized)
    {
        ACLLITE_LOG_WARNING("Live555Streamer already initialized");
        return false;
    }

    fH264Queue = queue;
    fQueueMutex = queueMutex;
    fQueueCond = queueCond;
    fRunning = running;
    fUseInternalQueue = false;
    fRtspPort = rtspPort;
    fStreamName = streamName;
    fFps = fps;

    ACLLITE_LOG_INFO(
        "Initializing Live555 RTSP server: port=%d, stream=%s, fps=%u",
        fRtspPort,
        fStreamName.c_str(),
        fFps);

    // 1. 创建任务调度器和环境
    fScheduler = BasicTaskScheduler::createNew();
    fEnv = BasicUsageEnvironment::createNew(*fScheduler);

    // 2. 创建 RTSP 服务器
    fRtspServer = RTSPServer::createNew(*fEnv, fRtspPort, nullptr);
    if (fRtspServer == nullptr)
    {
        ACLLITE_LOG_ERROR("Failed to create RTSP server: %s", fEnv->getResultMsg());
        return false;
    }

    // 3. 配置 RTP 包缓冲（必须在创建任何 RTP sink 之前设置）
    OutPacketBuffer::maxSize = 2000000; // 2MB RTP 包缓冲，防止大关键帧截断
    ACLLITE_LOG_INFO("OutPacketBuffer::maxSize set to %u bytes", OutPacketBuffer::maxSize);

    // 4. 创建服务器媒体会话
    fSms = ServerMediaSession::createNew(
        *fEnv,
        fStreamName.c_str(),
        fStreamName.c_str(),
        "H264 Video Stream from Hardware Encoder");

    // 5. 创建自定义 ServerMediaSubsession
    class H264LiveServerMediaSubsession : public OnDemandServerMediaSubsession
    {
      public:
        static H264LiveServerMediaSubsession *createNew(
            UsageEnvironment        &env,
            std::queue<H264Packet>  *queue,
            std::mutex              *queueMutex,
            std::condition_variable *queueCond,
            std::atomic<bool>       *running,
            unsigned                 fps)
        {
            return new H264LiveServerMediaSubsession(env, queue, queueMutex, queueCond, running, fps);
        }

      protected:
        H264LiveServerMediaSubsession(
            UsageEnvironment        &env,
            std::queue<H264Packet>  *queue,
            std::mutex              *queueMutex,
            std::condition_variable *queueCond,
            std::atomic<bool>       *running,
            unsigned                 fps)
            : OnDemandServerMediaSubsession(env, True),
              fQueue(queue),
              fQueueMutex(queueMutex),
              fQueueCond(queueCond),
              fRunning(running),
              fFps(fps)
        {
        }

        virtual FramedSource *createNewStreamSource(unsigned /*clientSessionId*/, unsigned &estBitrate) override
        {
            estBitrate = 4000; // kbps 估计（增加以适应高质量视频）
            Live555H264Source *source =
                Live555H264Source::createNew(envir(), fQueue, fQueueMutex, fQueueCond, fRunning, fFps);
            
            // 使用 framer 来正确解析 H.264 流
            // 数据源已实现分片发送，可以处理大关键帧
            return H264VideoStreamFramer::createNew(envir(), source);
        }

        virtual RTPSink *createNewRTPSink(
            Groupsock             *rtpGroupsock,
            unsigned char          rtpPayloadTypeIfDynamic,
            FramedSource * /*inputSource*/) override
        {
            return H264VideoRTPSink::createNew(envir(), rtpGroupsock, rtpPayloadTypeIfDynamic);
        }

      private:
        std::queue<H264Packet>  *fQueue;
        std::mutex              *fQueueMutex;
        std::condition_variable *fQueueCond;
        std::atomic<bool>       *fRunning;
        unsigned                 fFps;
    };

    // 6. 添加子会话
    fSms->addSubsession(
        H264LiveServerMediaSubsession::createNew(*fEnv, fH264Queue, fQueueMutex, fQueueCond, fRunning, fFps));

    // 7. 注册会话到 RTSP 服务器
    fRtspServer->addServerMediaSession(fSms);

    char *url = fRtspServer->rtspURL(fSms);
    ACLLITE_LOG_INFO("Live555 RTSP server initialized successfully");
    ACLLITE_LOG_INFO("Stream URL: %s", url);
    delete[] url;

    fInitialized = true;
    return true;
}

bool Live555Streamer::InitStandalone(int rtspPort, const std::string &streamName, unsigned fps)
{
    if (fInitialized)
    {
        ACLLITE_LOG_WARNING("Live555Streamer already initialized");
        return false;
    }

    // 绑定内部队列
    fH264Queue = &fInternalQueue;
    fQueueMutex = &fInternalMutex;
    fQueueCond = &fInternalCond;
    fRunning = &fInternalRunning;
    fInternalRunning = true;
    fUseInternalQueue = true;

    return Init(fH264Queue, fQueueMutex, fQueueCond, fRunning, rtspPort, streamName, fps);
}

void Live555Streamer::Start()
{
    if (!fInitialized)
    {
        ACLLITE_LOG_ERROR("Live555Streamer not initialized, call Init() first");
        return;
    }

    if (fEventLoopRunning)
    {
        ACLLITE_LOG_WARNING("Live555 event loop already running");
        return;
    }

    // 启动事件循环线程
    fEventLoopRunning = true;
    fEventLoopStopFlag.store(0);
    fEventLoopThread = std::thread(&Live555Streamer::EventLoopThread, this);

    ACLLITE_LOG_INFO("Live555 streamer started");
}

void Live555Streamer::EventLoopThread()
{
    ACLLITE_LOG_INFO("Live555 event loop thread started");
    fScheduler->doEventLoop(&fEventLoopStopFlag);
    fEventLoopRunning = false;
    ACLLITE_LOG_INFO("Live555 event loop thread stopped");
}

void Live555Streamer::Stop()
{
    if (!fInitialized)
    {
        return;
    }

    ACLLITE_LOG_INFO("Stopping Live555 streamer...");

    // 停止事件循环
    fEventLoopStopFlag.store(1);
    fEventLoopRunning = false;

    if (fEventLoopThread.joinable())
    {
        fEventLoopThread.join();
    }

    // 清理资源
    if (fRtspServer)
    {
        Medium::close(fRtspServer);
        fRtspServer = nullptr;
        fSms = nullptr; // 被 RTSPServer 管理
    }

    if (fEnv)
    {
        fEnv->reclaim();
        fEnv = nullptr;
    }

    if (fScheduler)
    {
        delete fScheduler;
        fScheduler = nullptr;
    }

    fInitialized = false;
    ACLLITE_LOG_INFO("Live555 streamer stopped");
}

std::string Live555Streamer::GetRtspUrl() const
{
    char url[256];
    snprintf(url, sizeof(url), "rtsp://%s:%d/%s", GetLocalHostIp().c_str(), fRtspPort, fStreamName.c_str());
    return std::string(url);
}

void Live555Streamer::Enqueue(const H264Packet &packet)
{
    if (!fInitialized || fH264Queue == nullptr || fQueueMutex == nullptr)
    {
        static int errCount = 0;
        if (++errCount % 100 == 1)
        {
            ACLLITE_LOG_WARNING("Enqueue called but not ready: init=%d, queue=%p, mutex=%p", 
                fInitialized, fH264Queue, fQueueMutex);
        }
        return;
    }
    
    {
        std::lock_guard<std::mutex> lock(*fQueueMutex);
        fH264Queue->push(packet);
        
        static int enqCount = 0;
        if (++enqCount % 100 == 1)
        {
            ACLLITE_LOG_INFO("Enqueued %d packets, current queue size: %zu, packet size: %zu bytes", 
                enqCount, fH264Queue->size(), packet.data.size());
        }
        
        // 简单限长，避免过度堆积（增大以容纳更多关键帧）
        const size_t MAX_Q = 300; // 约 12s@25fps
        while (fH264Queue->size() > MAX_Q)
        {
            fH264Queue->pop();
        }
    }
    if (fQueueCond)
        fQueueCond->notify_one();
}

#else // !USE_LIVE555

// 空实现,避免链接错误
Live555Streamer::Live555Streamer()
    : fScheduler(nullptr),
      fEnv(nullptr),
      fRtspServer(nullptr),
      fSms(nullptr),
      fRtspPort(0),
      fFps(0),
      fH264Queue(nullptr),
      fQueueMutex(nullptr),
      fQueueCond(nullptr),
      fRunning(nullptr),
      fEventLoopRunning(false),
      fInitialized(false)
{
    fEventLoopStopFlag.store(0);
    ACLLITE_LOG_ERROR("Live555Streamer: USE_LIVE555 not defined, this class is disabled");
}

Live555Streamer::~Live555Streamer() {}

bool Live555Streamer::Init(
    std::queue<H264Packet> *,
    std::mutex *,
    std::condition_variable *,
    std::atomic<bool> *,
    int,
    const std::string &,
    unsigned)
{
    ACLLITE_LOG_ERROR("Live555Streamer: USE_LIVE555 not defined");
    return false;
}

void Live555Streamer::Start() { ACLLITE_LOG_ERROR("Live555Streamer: USE_LIVE555 not defined"); }

void Live555Streamer::Stop() {}

std::string Live555Streamer::GetRtspUrl() const { return "rtsp://disabled"; }

void Live555Streamer::EventLoopThread() {}

#endif // USE_LIVE555
