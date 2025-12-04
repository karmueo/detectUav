# MixFormerV2 OM 模型推理实现

本目录包含使用华为昇腾（Ascend）OM 模型进行 MixFormerV2 目标跟踪推理的实现。

## 文件说明

- `baseTrack_om.h/cpp`: 基础跟踪类，包含图像处理和后处理函数
- `mixformerv2_om.h/cpp`: MixFormerV2 跟踪器实现，使用 AclLiteModel 进行 OM 模型推理

## 模型输入输出规格

### 输入
1. **template**: 维度 `[1, 3, 112, 112]`, float32
2. **online_template**: 维度 `[1, 3, 112, 112]`, float32
3. **search**: 维度 `[1, 3, 224, 224]`, float32

### 输出
1. **pred_boxes**: 维度 `[1, 1, 4]`, float32 - 预测的边界框 [cx, cy, w, h]
2. **pred_scores**: 维度 `[1]`, float32 - 预测置信度分数

## 使用方法

```cpp
#include "mixformerv2_om.h"
#include "AclLiteApp.h"

// 1. 初始化 ACL 资源
AclLiteError ret = AclLiteResource::Instance().Init();
if (ret != ACLLITE_OK) {
    // 错误处理
}

// 2. 创建跟踪器实例
std::string model_path = "model/mixformerv2_online_small.om";
MixformerV2OM tracker(model_path);

// 3. 初始化模型
if (tracker.InitModel() != 0) {
    // 错误处理
}

// 4. 初始化跟踪（使用第一帧和初始边界框）
cv::Mat first_frame = cv::imread("first_frame.jpg");
DrOBB init_bbox = {
    {x0, y0, x1, y1, w, h, cx, cy},  // DrBBox
    1.0f,  // score
    0      // class_id
};
if (tracker.init(first_frame, init_bbox) != 0) {
    // 错误处理
}

// 5. 在后续帧中跟踪
cv::Mat current_frame = cv::imread("current_frame.jpg");
const DrOBB& result = tracker.track(current_frame);

// 6. 使用跟踪结果
float x0 = result.box.x0;
float y0 = result.box.y0;
float x1 = result.box.x1;
float y1 = result.box.y1;
float score = result.score;

// 7. 清理资源
AclLiteResource::Instance().Release();
```

## 配置参数

可以通过以下方法设置跟踪参数：

```cpp
tracker.setTemplateSize(112);              // 模板图像大小
tracker.setSearchSize(224);                // 搜索图像大小
tracker.setTemplateFactor(2.0f);           // 模板区域因子
tracker.setSearchFactor(5.0f);             // 搜索区域因子
tracker.setUpdateInterval(200);            // 在线模板更新间隔
tracker.setTemplateUpdateScoreThreshold(0.85f);  // 模板更新分数阈值
tracker.setMaxScoreDecay(0.98f);           // 最大分数衰减率
```

## 注意事项

1. 确保 OM 模型的输入输出维度与代码中的定义一致
2. 输入图像需要是 BGR 格式的 cv::Mat
3. 边界框坐标使用像素坐标（左上角为原点）
4. 模型路径需要是有效的 OM 模型文件路径

## 依赖

- OpenCV
- AclLite (华为昇腾 ACL Lite 框架)
- 昇腾 ACL 库
