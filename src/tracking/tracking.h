#ifndef TRACKING_H
#define TRACKING_H

#include "AclLiteModel.h"
#include "AclLiteThread.h"
#include "Params.h"
#include <memory>
#include <opencv2/opencv.hpp>
#include <string>
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
     * @param model_path 输入：OM 模型文件路径
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
     * @brief 设置模板输入数据
     * @param data 输入：数据指针
     * @param size 输入：数据大小（元素个数）
     */
    void setInputTemplateData(const float *data, size_t size);

    /**
     * @brief 设置在线模板输入数据
     * @param data 输入：数据指针
     * @param size 输入：数据大小（元素个数）
     */
    void setInputOnlineTemplateData(const float *data, size_t size);

    /**
     * @brief 设置搜索输入数据
     * @param data 输入：数据指针
     * @param size 输入：数据大小（元素个数）
     */
    void setInputSearchData(const float *data, size_t size);

    /**
     * @brief 执行推理
     */
    void infer();

    /**
     * @brief 获取预测框输出
     * @return 预测框数据向量（大小为4）
     */
    std::vector<float> getOutputPredBoxes() const;

    /**
     * @brief 获取预测分数输出
     * @return 预测分数数据向量（大小为1）
     */
    std::vector<float> getOutputPredScores() const;
    
    /**
     * @brief 发送跟踪状态反馈给DataInput线程
     * @param detectDataMsg 输入：检测数据消息
     */
    void SendTrackingStateFeedback(std::shared_ptr<DetectDataMsg> detectDataMsg);
    
    private:
    /**
     * @brief 重置最大预测分数
     */
    void resetMaxPredScore();

    AclLiteError MsgSend(std::shared_ptr<DetectDataMsg> detectDataMsg);

    /// 归一化常量
    static const float mean_vals[3]; ///< 均值常量
    static const float norm_vals[3]; ///< 方差常量

    /**
     * @brief 从图像采样目标区域
     * @param img 输入：输入图像
     * @param patch 输出：采样后的图像块
     * @param bbox 输入：边界框
     * @param factor 输入：搜索区域因子
     * @param output_size 输入：输出大小
     * @param resize_factor 输出：缩放因子
     * @return 成功返回 0，失败返回 -1
     */
    int sample_target(
        const cv::Mat &img,
        cv::Mat       &patch,
        const DrBBox  &bbox,
        float          factor,
        int            output_size,
        float         &resize_factor);

    /**
     * @brief 对图像进行半归一化处理
     * @param img 输入：输入图像
     * @param output 输出：归一化后的数据
     */
    void half_norm(const cv::Mat &img, float *output);

    /**
     * @brief 计算边界框
     * @param pred_boxes 输入：预测框数据
     * @param resize_factor 输入：缩放因子
     * @param search_size 输入：搜索大小
     * @return 计算后的边界框
     */
    DrBBox
    cal_bbox(const float *pred_boxes, float resize_factor, int search_size);

    /**
     * @brief 将边界框映射回原图坐标
     * @param box 输入输出：边界框
     * @param resize_factor 输入：缩放因子
     * @param search_size 输入：搜索大小
     */
    void map_box_back(DrBBox &box, float resize_factor, int search_size);

    /**
     * @brief 裁剪边界框到图像边界
     * @param box 输入输出：边界框
     * @param img_h 输入：图像高度
     * @param img_w 输入：图像宽度
     * @param border 输入：边界边距
     */
    void clip_box(DrBBox &box, int img_h, int img_w, int border);

    /// 输入缓冲区大小（元素数，非字节）
    size_t input_template_size;        ///< 模板输入缓冲区大小
    size_t input_online_template_size; ///< 在线模板输入缓冲区大小
    size_t input_search_size;          ///< 搜索输入缓冲区大小
    size_t output_pred_boxes_size;     ///< 预测框输出缓冲区大小
    size_t output_pred_scores_size;    ///< 预测分数输出缓冲区大小

    /// 主机内存缓冲区
    float *input_template;        ///< 模板输入数据
    float *input_online_template; ///< 在线模板输入数据
    float *input_search;          ///< 搜索输入数据
    float *output_pred_boxes;     ///< 预测框输出数据
    float *output_pred_scores;    ///< 预测分数输出数据

    /// 模板大小配置
    int   template_size;   ///< 模板大小
    int   search_size;     ///< 搜索大小
    float template_factor; ///< 模板因子
    float search_factor;   ///< 搜索因子

    /// 跟踪状态
    DrOBB              object_box;                      ///< 对象边界框
    DrBBox             state;                           ///< 当前状态
    int                frame_id;                        ///< 帧 ID
    float              max_pred_score;                  ///< 最大预测分数
    int                update_interval;                 ///< 更新间隔
    float              template_update_score_threshold; ///< 模板更新分数阈值
    float              max_score_decay;                 ///< 最大分数衰减
    std::vector<float> new_online_template;             ///< 新在线模板

    /// OM 模型
    AclLiteModel model_;             ///< ACL 模型
    std::string  model_path_;        ///< 模型路径
    bool         model_initialized_; ///< 模型是否已初始化

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
};

#endif // TRACKING_H
