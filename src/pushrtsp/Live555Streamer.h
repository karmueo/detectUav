#pragma once

#include "pictortsp.h"
#include <atomic>
#include <condition_variable>
#include <mutex>
#include <queue>
#include <string>
#include <thread>
#include <vector>

// 前向声明 live555 类型(避免直接包含头文件导致编译错误)
class TaskScheduler;
class UsageEnvironment;
class RTSPServer;
class ServerMediaSession;
class FramedSource;

// 自定义 H264 数据源,从队列中读取编码数据
// 注意: 实现在 .cpp 文件中,需要包含完整的 live555 头文件
class Live555H264Source;

// Live555 RTSP 推流器
// 功能: 直接从 g_h264Queue 读取编码数据并通过 RTSP 推流
class Live555Streamer
{
  public:
    Live555Streamer();
    ~Live555Streamer();

    // 初始化 RTSP 服务器(使用外部队列)
    bool Init(
        std::queue<H264Packet>  *queue,
        std::mutex              *queueMutex,
        std::condition_variable *queueCond,
        std::atomic<bool>       *running,
        int                      rtspPort,
        const std::string       &streamName,
        unsigned                 fps = 25);

    // 初始化 RTSP 服务器(使用内部自带队列)
    bool InitStandalone(int rtspPort, const std::string &streamName, unsigned fps = 25);

    // 启动 RTSP 服务器事件循环(在独立线程中运行)
    void Start();

    // 停止 RTSP 服务器
    void Stop();

    // 获取 RTSP URL
    std::string GetRtspUrl() const;

    // 检查是否正在运行
    bool IsRunning() const { return fInitialized && fEventLoopRunning; }

    // 直接投喂一帧 H264 数据(仅在使用内部队列或提供的队列有效)
    void Enqueue(const H264Packet &packet);

  private:
    void EventLoopThread();

    // Live555 核心对象(使用前向声明的类型)
    TaskScheduler      *fScheduler;
    UsageEnvironment   *fEnv;
    RTSPServer         *fRtspServer;
    ServerMediaSession *fSms;

    // 配置参数
    int         fRtspPort;
    std::string fStreamName;
    unsigned    fFps;

    // 队列选择: 可使用外部队列或内部队列
    // 外部队列引用(不拥有所有权)
    std::queue<H264Packet>  *fH264Queue;
    std::mutex              *fQueueMutex;
    std::condition_variable *fQueueCond;
    std::atomic<bool>       *fRunning;

    // 内部队列(当未提供外部队列时启用)
    std::queue<H264Packet>  fInternalQueue;
    std::mutex              fInternalMutex;
    std::condition_variable fInternalCond;
    std::atomic<bool>       fInternalRunning;
    bool                    fUseInternalQueue;

    // 事件循环控制
    std::thread       fEventLoopThread;
    std::atomic<char> fEventLoopStopFlag;
    std::atomic<bool> fEventLoopRunning;
    bool              fInitialized;
};
