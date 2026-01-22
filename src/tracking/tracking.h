#ifndef TRACKING_H
#define TRACKING_H

#include "AclLiteModel.h"
#include "AclLiteThread.h"
#include "Params.h"
#include <array>
#include <memory>
#include <opencv2/opencv.hpp>
#include <string>
#include <utility>
#include <vector>

struct DrBBox
{
    float x0; ///< 左上角 x 坐标
    float y0; ///< 左上角 y 坐标
    float x1; ///< 右下角 x 坐标
    float y1; ///< 右下角 y 坐标
    float w;  ///< 宽度
    float h;  ///< 高度
    float cx; ///< 中心 x 坐标
    float cy; ///< 中心 y 坐标
};

struct DrOBB
{
    DrBBox box;      ///< 边界框
    float  score;    ///< 置信度分数
    int    class_id; ///< 类别 ID
    float  initScore; ///< 初始化检测时的置信度
};

class Tracking : public AclLiteThread
{
  public:
    /**
     * @brief 构造函数
     * @param model_path 输入：Nanotrack 模型路径，支持单个 head 路径或“head;backbone;search”三段配置
     */
    Tracking(const std::string &model_path);

    /**
     * @brief 析构函数
     */
    ~Tracking();

    /**
     * @brief 初始化模型
     * @return 成功返回 0，失败返回非 0
     */
    int InitModel();

    /**
     * @brief AclLiteThread 初始化重写
     * @return AclLiteError 错误码
     */
    AclLiteError Init();

    /**
     * @brief AclLiteThread 消息处理重写
     * @param msgId 输入：消息 ID
     * @param data 输入：消息数据
     * @return AclLiteError 错误码
     */
    AclLiteError Process(int msgId, std::shared_ptr<void> data);

    /**
     * @brief 初始化跟踪器
     * @param img 输入：模板图像
     * @param bbox 输入：初始边界框
     * @return 成功返回 0，失败返回非 0
     */
    int init(const cv::Mat &img, DrOBB bbox);

    /**
     * @brief 在当前帧跟踪对象
     * @param img 输入：当前帧图像
     * @return 跟踪结果边界框的常量引用
     */
    const DrOBB &track(const cv::Mat &img);

    /**
     * @brief 设置模板大小
     * @param size 输入：模板大小
     */
    void setTemplateSize(int size);

    /**
     * @brief 设置搜索大小
     * @param size 输入：搜索大小
     */
    void setSearchSize(int size);

    /**
     * @brief 设置模板因子
     * @param factor 输入：模板因子
     */
    void setTemplateFactor(float factor);

    /**
     * @brief 设置搜索因子
     * @param factor 输入：搜索因子
     */
    void setSearchFactor(float factor);

    /**
     * @brief 设置更新间隔
     * @param interval 输入：更新间隔
     */
    void setUpdateInterval(int interval);

    /**
     * @brief 设置模板更新分数阈值
     * @param threshold 输入：分数阈值
     */
    void setTemplateUpdateScoreThreshold(float threshold);

    /**
     * @brief 设置最大分数衰减
     * @param decay 输入：衰减值
     */
    void setMaxScoreDecay(float decay);
    
    /**
     * @brief 设置保持跟踪的最低置信度阈值
     * @param threshold 输入：置信度阈值
     */
    void setConfidenceActiveThreshold(float threshold);
    
    /**
     * @brief 设置触发重新检测的置信度阈值
     * @param threshold 输入：置信度阈值
     */
    void setConfidenceRedetectThreshold(float threshold);
    
    /**
     * @brief 设置最大连续跟踪丢失帧数
     * @param maxFrames 输入：最大帧数
     */
    void setMaxTrackLossFrames(int maxFrames);

    /**
     * @brief 设置是否启用检测验证跟踪
     * @param enabled 输入：true 启用，false 关闭
     */
    void setTrackingValidationEnabled(bool enabled);

    /**
     * @brief 设置检测验证的 IOU 阈值
     * @param threshold 输入：IOU 阈值
     */
    void setTrackingValidationIouThreshold(float threshold);

    /**
     * @brief 设置检测验证的最大错误次数
     * @param maxErrors 输入：最大错误次数
     */
    void setTrackingValidationMaxErrors(int maxErrors);

    /**
     * @brief 是否启用可疑静止目标过滤
     * @param enabled 输入：true 启用，false 关闭
     */
    void setStaticTargetFilterEnabled(bool enabled);

    /**
     * @brief 设置中心点判定阈值（像素）
     * @param threshold 输入：阈值（像素）
     */
    void setStaticCenterThreshold(float threshold);

    /**
     * @brief 设置尺寸判定阈值（像素）
     * @param threshold 输入：阈值（像素）
     */
    void setStaticSizeThreshold(float threshold);

    /**
     * @brief 设置静止判定连续帧数
     * @param frames 输入：连续帧数
     */
    void setStaticFrameThreshold(int frames);

    /**
     * @brief 发送跟踪状态反馈给DataInput线程
     * @param detectDataMsg 输入：检测数据消息
     */
    void SendTrackingStateFeedback(std::shared_ptr<DetectDataMsg> detectDataMsg);
    
    private:
    AclLiteError MsgSend(std::shared_ptr<DetectDataMsg> detectDataMsg);

    /**
     * @brief 初始化 Nanotrack 模型路径
     * @param model_path 输入：配置中的模型路径字符串
     */
    void InitNanotrackModelPath(const std::string &model_path);

    /**
     * @brief 初始化 Nanotrack 模型输入输出尺寸
     * @return 成功返回 0，失败返回 -1
     */
    int InitNanotrackModelIO();

    /**
     * @brief 运行模板 Backbone 推理
     * @param input 输入：模板图像 CHW 数据
     * @param out_shape 输出：特征张量形状
     * @return 输出特征向量
     */
    std::vector<float> RunBackbone(const std::vector<float> &input,
                                   std::vector<int64_t> &out_shape);

    /**
     * @brief 运行搜索 Backbone 推理
     * @param input 输入：搜索图像 CHW 数据
     * @param out_shape 输出：特征张量形状
     * @return 输出特征向量
     */
    std::vector<float> RunSearchBackbone(const std::vector<float> &input,
                                         std::vector<int64_t> &out_shape);

    /**
     * @brief 运行 Head 推理
     * @param zf 输入：模板特征
     * @param zf_shape 输入：模板特征形状
     * @param xf 输入：搜索特征
     * @param xf_shape 输入：搜索特征形状
     * @param cls_shape 输出：分类输出形状
     * @param loc_shape 输出：回归输出形状
     * @return 分类与回归输出
     */
    std::pair<std::vector<float>, std::vector<float>>
    RunHead(const std::vector<float> &zf,
            const std::vector<int64_t> &zf_shape,
            const std::vector<float> &xf,
            const std::vector<int64_t> &xf_shape,
            std::vector<int64_t> &cls_shape,
            std::vector<int64_t> &loc_shape);

    /**
     * @brief 确保 score 尺寸并更新窗口与点
     * @param size 输入：score 尺寸
     */
    void EnsureScoreSize(int size);

    /**
     * @brief 构建汉宁窗
     * @param size 输入：窗口尺寸
     * @return 窗口数据
     */
    std::vector<float> BuildWindow(int size);

    /**
     * @brief 构建特征点坐标
     * @param stride 输入：步长
     * @param size 输入：score 尺寸
     * @return 点坐标列表
     */
    std::vector<cv::Point2f> BuildPoints(int stride, int size);

    /**
     * @brief 裁剪并缩放子图
     * @param img 输入：原图
     * @param pos 输入：中心位置
     * @param model_sz 输入：模型输入尺寸
     * @param original_sz 输入：裁剪尺寸
     * @param avg_chans 输入：均值
     * @return CHW 数据
     */
    std::vector<float> GetSubwindow(const cv::Mat &img,
                                    const cv::Point2f &pos,
                                    int model_sz,
                                    int original_sz,
                                    const cv::Scalar &avg_chans);

    /**
     * @brief 对齐特征图尺寸
     * @param feat 输入：特征数据
     * @param shape 输入：特征形状
     * @param target_hw 输入：目标高宽
     * @param out_shape 输出：对齐后形状
     * @return 对齐后特征
     */
    std::vector<float> AlignFeature(const std::vector<float> &feat,
                                    const std::vector<int64_t> &shape,
                                    const std::pair<int, int> &target_hw,
                                    std::vector<int64_t> &out_shape);

    /**
     * @brief 解析分类得分
     * @param cls 输入：分类输出
     * @param shape 输入：分类形状
     * @return 得分向量
     */
    std::vector<float> ConvertScore(const std::vector<float> &cls,
                                    const std::vector<int64_t> &shape) const;

    /**
     * @brief 解析回归框
     * @param loc 输入：回归输出
     * @param shape 输入：回归形状
     * @return bbox 向量
     */
    std::vector<float> ConvertBBox(const std::vector<float> &loc,
                                   const std::vector<int64_t> &shape) const;

    /**
     * @brief 边界框裁剪
     * @param cx 输入：中心 x
     * @param cy 输入：中心 y
     * @param width 输入：宽度
     * @param height 输入：高度
     * @param rows 输入：图像高
     * @param cols 输入：图像宽
     * @return 裁剪后参数
     */
    std::array<float, 4> BboxClip(float cx, float cy, float width,
                                  float height, int rows, int cols) const;

    /**
     * @brief 从输出维度构造形状向量
     * @param dims 输入：ACL 输出维度
     * @return 形状向量
     */
    std::vector<int64_t> DimsToShape(const aclmdlIODims &dims) const;

    /**
     * @brief 从输入元素数推导方形高宽
     * @param elements 输入：元素数
     * @param channels 输入：通道数
     * @return 高宽对
     */
    std::pair<int, int> CalcSquareHW(size_t elements, int channels) const;

    /**
     * @brief 计算两个检测框的 IOU
     * @param a 输入：检测框 A
     * @param b 输入：检测框 B
     * @return IOU 值，范围 [0,1]
     */
    float ComputeIou(const DetectionOBB &a, const DetectionOBB &b) const;

    bool IsBlockedDetection(const DetectionOBB &det) const;
    bool UpdateStaticTrackingState(const DrBBox &box);
    void FillStaticFilterState(std::shared_ptr<DetectDataMsg> &msg) const;

    struct TrackerConfig
    {
        int   exemplar_size = 127;     ///< 模板尺寸
        int   instance_size = 255;     ///< 搜索尺寸
        int   score_size = 15;         ///< score 尺寸
        int   stride = 16;             ///< 步长
        float context_amount = 0.5f;   ///< 上下文比例
        float window_influence = 0.455f; ///< 窗口影响
        float penalty_k = 0.138f;      ///< 惩罚系数
        float lr = 0.348f;             ///< 学习率
    };

    TrackerConfig cfg_; ///< 跟踪器配置

    /// 模型 I/O 尺寸（元素数，非字节）
    size_t backbone_input_size_ = 0;      ///< backbone 输入元素数
    size_t backbone_output_size_ = 0;     ///< backbone 输出元素数
    size_t search_input_size_ = 0;        ///< search backbone 输入元素数
    size_t search_output_size_ = 0;       ///< search backbone 输出元素数
    size_t head_input_z_size_ = 0;        ///< head 模板输入元素数
    size_t head_input_x_size_ = 0;        ///< head 搜索输入元素数
    size_t head_output_cls_size_ = 0;     ///< head 分类输出元素数
    size_t head_output_loc_size_ = 0;     ///< head 回归输出元素数

    /// 模型输出缓存
    std::vector<float> backbone_output_; ///< backbone 输出缓存
    std::vector<float> search_output_;   ///< search 输出缓存
    std::vector<float> head_output_cls_; ///< head 分类输出缓存
    std::vector<float> head_output_loc_; ///< head 回归输出缓存

    /// 模型输出形状
    std::vector<int64_t> backbone_output_shape_; ///< backbone 输出形状
    std::vector<int64_t> search_output_shape_;   ///< search 输出形状
    std::vector<int64_t> head_cls_shape_;        ///< head 分类形状
    std::vector<int64_t> head_loc_shape_;        ///< head 回归形状

    /// 特征图对齐尺寸
    std::pair<int, int> head_template_hw_{-1, -1}; ///< head 模板高宽
    std::pair<int, int> head_search_hw_{-1, -1};   ///< head 搜索高宽
    std::pair<int, int> template_input_hw_{-1, -1}; ///< 模板输入高宽
    std::pair<int, int> search_input_hw_{-1, -1};   ///< 搜索输入高宽

    /// Nanotrack 状态
    std::vector<float> window_;          ///< 汉宁窗
    std::vector<cv::Point2f> points_;    ///< 特征点坐标
    cv::Point2f center_pos_{0.f, 0.f};   ///< 目标中心
    cv::Point2f size_{0.f, 0.f};         ///< 目标尺寸
    cv::Scalar channel_average_;         ///< 通道均值
    std::vector<float> zf_;              ///< 模板特征
    std::vector<int64_t> zf_shape_;      ///< 模板特征形状
    std::vector<int64_t> subwindow_shape_; ///< 子图形状
    float last_score_ = 0.f;             ///< 上次得分
    float search_scale_factor_ = 1.0f;   ///< 搜索缩放因子

    /// 模型路径
    std::string head_model_path_;        ///< head 模型路径
    std::string backbone_model_path_;    ///< backbone 模型路径
    std::string search_model_path_;      ///< search backbone 模型路径

    /// OM 模型
    AclLiteModel head_model_;            ///< head 模型
    AclLiteModel backbone_model_;        ///< backbone 模型
    AclLiteModel search_model_;          ///< search backbone 模型
    bool         has_search_backbone_ = false; ///< 是否存在独立 search backbone

    int head_input_z_index_ = 0;        ///< head 模板输入索引
    int head_input_x_index_ = 1;        ///< head 搜索输入索引
    int head_output_cls_index_ = 0;      ///< head 分类输出索引
    int head_output_loc_index_ = 1;      ///< head 回归输出索引

    /// 跟踪状态
    DrOBB              object_box;                      ///< 对象边界框
    DrBBox             state;                           ///< 当前状态
    int                frame_id;                        ///< 帧 ID
    int                update_interval;                 ///< 更新间隔
    float              template_update_score_threshold; ///< 模板更新分数阈值
    float              max_score_decay;                 ///< 最大分数衰减
    bool               model_initialized_;              ///< 模型是否已初始化

    /// 线程/上下文
    aclrtRunMode runMode_;                      ///< 运行模式
    int          dataOutputThreadId_ = -1;      ///< 数据输出线程 ID
    int          dataInputThreadId_ = -1;       ///< 数据输入线程 ID (用于状态反馈)
    bool         tracking_initialized_ = false; ///< 跟踪是否已初始化
    
    /// ============ 跟踪状态管理 ============
    float        confidence_active_threshold_ = 0.70f;    ///< 保持跟踪的最低置信度
    float        confidence_redetect_threshold_ = 0.40f;  ///< 触发重新检测的置信度
    int          max_track_loss_frames_ = 10;             ///< 连续丢失帧数阈值
    int          track_loss_count_ = 0;                   ///< 当前连续丢失帧数
    float        current_tracking_confidence_ = 0.0f;     ///< 当前跟踪置信度

    /// ============ 可疑静止目标过滤 ============
    bool   filter_static_target_ = false;     ///< 是否启用静止目标过滤
    float  static_center_threshold_ = 2.0f;   ///< 中心点变化阈值（像素）
    float  static_size_threshold_ = 2.0f;     ///< 尺寸变化阈值（像素）
    int    static_frame_threshold_ = 30;      ///< 连续静止帧数阈值
    int    static_frame_count_ = 0;           ///< 当前连续静止帧计数
    bool   has_last_box_ = false;             ///< 是否已有上一帧框
    DrBBox last_box_ = {0, 0, 0, 0, 0, 0, 0, 0};
    bool   has_blocked_target_ = false;       ///< 是否存在已阻断目标
    DrBBox blocked_target_ = {0, 0, 0, 0, 0, 0, 0, 0};

    /// ============ 检测验证跟踪 ============
    bool   tracking_validation_enabled_ = false; ///< 是否启用检测验证跟踪
    float  tracking_validation_iou_threshold_ = 0.3f; ///< IOU 阈值
    int    tracking_validation_max_errors_ = 3; ///< 最大错误次数
    int    tracking_validation_error_count_ = 0; ///< 当前错误次数
};

#endif // TRACKING_H
