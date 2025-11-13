/**
 * @file VencCallback.h
 * 
 * 用于RTSP推流的视频编码回调接口
 */
#ifndef VENC_CALLBACK_H
#define VENC_CALLBACK_H

#include <functional>
#include <cstdint>

// 编码数据回调函数类型
// 参数: data - 编码后的H264数据指针
//       size - 数据大小
//       userData - 用户自定义数据
typedef std::function<void(void* data, uint32_t size, void* userData)> VencDataCallback;

#endif // VENC_CALLBACK_H
