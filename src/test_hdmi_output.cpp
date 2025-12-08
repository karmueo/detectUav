#include "AclLiteResource.h"
#include "Params.h"
#include "hdmiOutput/hdmiOutputThread.h"
#include <cstring>
#include <iostream>
#include <memory>
#include <opencv2/opencv.hpp>
#include <string>
#include <unistd.h>

using std::shared_ptr;

namespace
{
const uint32_t kHdmiWidth = 1920;
const uint32_t kHdmiHeight = 1080;
}

// Simple HDMI smoke test: load an image, convert to NV12 1080p and display it
// via HdmiOutputThread.
int main(int argc, char **argv)
{
    if (argc < 2)
    {
        std::cerr << "Usage: test_hdmi_output <image_path>" << std::endl;
        return 1;
    }

    std::string imagePath = argv[1];

    AclLiteResource aclDev;
    if (aclDev.Init() != ACLLITE_OK)
    {
        std::cerr << "Init ACL resources failed" << std::endl;
        return 1;
    }

    cv::Mat bgr = cv::imread(imagePath);
    if (bgr.empty())
    {
        std::cerr << "Failed to read image: " << imagePath << std::endl;
        return 1;
    }

    cv::Mat resized;
    cv::resize(bgr, resized, cv::Size(kHdmiWidth, kHdmiHeight));

    cv::Mat yuv420;
    cv::cvtColor(resized, yuv420, cv::COLOR_BGR2YUV_I420);

    size_t ySize = static_cast<size_t>(kHdmiWidth) * kHdmiHeight;
    size_t uvPlaneSize = ySize / 4; // I420 has separate U/V planes of size /4
    size_t nv12Size = ySize + uvPlaneSize * 2;

    uint8_t *nv12Buf = new (std::nothrow) uint8_t[nv12Size];
    if (nv12Buf == nullptr)
    {
        std::cerr << "Failed to allocate NV12 buffer" << std::endl;
        return 1;
    }

    // Copy Y plane
    std::memcpy(nv12Buf, yuv420.data, ySize);
    // Interleave U/V planes into NV12
    uint8_t *srcU = yuv420.data + ySize;
    uint8_t *srcV = srcU + uvPlaneSize;
    uint8_t *dstUV = nv12Buf + ySize;
    for (size_t i = 0; i < uvPlaneSize; ++i)
    {
        dstUV[2 * i] = srcU[i];
        dstUV[2 * i + 1] = srcV[i];
    }

    ImageData image;
    image.format = PIXEL_FORMAT_YUV_SEMIPLANAR_420;
    image.width = kHdmiWidth;
    image.height = kHdmiHeight;
    image.alignWidth = kHdmiWidth;
    image.alignHeight = kHdmiHeight;
    image.size = static_cast<uint32_t>(nv12Size);
    image.data = SHARED_PTR_U8_BUF(nv12Buf);

    HdmiOutputThread hdmiThread(aclDev.GetRunMode());
    if (hdmiThread.Init() != ACLLITE_OK)
    {
        std::cerr << "HDMI init failed" << std::endl;
        return 1;
    }

    auto detectMsg = std::make_shared<DetectDataMsg>();
    detectMsg->decodedImg.push_back(image);
    detectMsg->isLastFrame = false; // avoid extra SendMessage in test path

    AclLiteError ret = hdmiThread.Process(MSG_HDMI_DISPLAY, detectMsg);
    if (ret != ACLLITE_OK)
    {
        std::cerr << "Send frame to HDMI failed, error " << ret << std::endl;
        return 1;
    }

    std::cout << "Frame pushed to HDMI. Press Enter after checking the display..."
              << std::endl;
    std::string line;
    std::getline(std::cin, line);
    return 0;
}
