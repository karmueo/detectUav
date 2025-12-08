/**
* Copyright (c) Huawei Technologies Co., Ltd.
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*
* HDMI output thread: display NV12 frames via VO/HDMI.
*/
#ifndef HDMI_OUTPUT_THREAD_H
#define HDMI_OUTPUT_THREAD_H
#pragma once

#include "AclLiteImageProc.h"
#include "AclLiteThread.h"
#include "AclLiteUtils.h"
#include "Params.h"
#include "acl/acl.h"
extern "C" {
#include "hi_common_vo.h"
#include "hi_media_common.h"
#include "hi_mpi_hdmi.h"
#include "hi_mpi_sys.h"
#include "hi_mpi_vo.h"
}

#define VO_MST_ALIGN_16 16
#define VO_MST_ALIGN_2 2
#define VO_ALIGN_BACK(x, a) ((a) * (((x) / (a))))
#define VO_TEST_ALIGN_BACK(x, a) (((a) * ((((x) + (a) - 1) / (a)))))
#define VO_LAYER_VHD0 0
#define DEV_DHD0 0
#define VO_CHECK_RET(express, name)                                                              \
    do {                                                                                         \
        hi_s32 _ret;                                                                             \
        _ret = (express);                                                                        \
        if (_ret != HI_SUCCESS) {                                                                \
            ACLLITE_LOG_ERROR("%s failed at %s: LINE: %d with %#x!\n", name, __FUNCTION__, __LINE__, _ret); \
        }                                                                                        \
    } while (0)

typedef struct
{
    hi_vo_intf_sync intf_sync;
    hi_char        *name;
    hi_u32          width;
    hi_u32          height;
    hi_u32          frame_rate;
} vo_mst_sync_info;

typedef enum {
    VB_REMAP_MODE_NONE = 0,    // no remap
    VB_REMAP_MODE_NOCACHE = 1, // no cache remap
    VB_REMAP_MODE_CACHED = 2,  // cache remap
    VB_REMAP_MODE_BUTT
} vo_vb_remap_mode;

typedef struct {
    hi_u64          blk_size;
    hi_u32          blk_cnt;
    vo_vb_remap_mode remap_mode;
} hi_vb_pool_config;

class HdmiOutputThread : public AclLiteThread
{
  public:
    HdmiOutputThread(aclrtRunMode runMode, VencConfig vencConfig = VencConfig());
    ~HdmiOutputThread();

    AclLiteError Init() override;
    AclLiteError Process(int msgId, std::shared_ptr<void> msgData) override;

  private:
    AclLiteError InitHdmi();
    void         DeinitHdmi();
    AclLiteError HandleDisplay(std::shared_ptr<DetectDataMsg> detectDataMsg);
    AclLiteError DisplayFrame(const ImageData &image);
    AclLiteError EnsureImageOnHost(const ImageData &deviceImg, ImageData &hostImg);

  private:
    aclrtRunMode runMode_;
    VencConfig   vencConfig_;
    hi_u32       vbPoolVal_;
    vo_mst_sync_info syncInfo_;
    bool         sysInited_;
    bool         hdmiInited_;
    hi_s32       devId_;
    hi_s32       layerId_;
    hi_vo_intf_type intfType_;
    hi_vo_intf_sync intfSync_;
};

#endif
