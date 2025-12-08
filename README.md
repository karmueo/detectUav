# detectUav

基于 Ascend ACL 的无人机检测与单目标跟踪管线。加载检测模型和可选跟踪模型，拉取 RTSP 等输入并将标注后的 H.264 通过 RTSP 推流输出。

## 构建与运行
1. 安装 Ascend 驱动/工具链，设置 `SYSROOT`、`DDK_PATH`、`NPU_HOST_LIB`（缺省 `/usr/local/Ascend/ascend-toolkit/latest`）。若需要 RTSP 推流，确保 FFmpeg/OpenCV 与 live555 可用。
2. 配置并编译（Ninja 示例）：
   ```bash
   mkdir -p build && cd build
   cmake -G Ninja ..
   cmake --build . --target main test_mixformerv2_om
   ```
3. 从 `build/` 目录使用 JSON 配置运行：
   ```bash
   ./src/out/main ../scripts/test.json
   ```

## JSON 配置说明
程序通过 JSON 描述设备、模型与 IO。样例见 `scripts/config.json`、`scripts/hk_rtsp.json`、`scripts/test.json`。

顶层结构：
```json
{
  "device_config": [
    {
      "device_id": 0,
      "model_config": [
        {
          "model_path": "./model/yolov11n_110_rgb_640_raw_v0_bs1.om",
          "model_width": 640,
          "model_heigth": 640,
          "model_batch": 1,
          "frame_decimation": 0,
          "track_config": { ... },
          "io_info": [ { ... 每路流的配置 ... } ]
        }
      ]
    }
  ]
}
```

### 字段速查
- `device_config[]`：每个条目对应一块 Ascend 设备。
  - `device_id`：设备编号。
  - `model_config[]`：该设备上的检测模型列表。
    - `model_path`：检测 `.om` 路径（相对路径从运行目录解析）。
    - `model_width` / `model_heigth`：模型输入宽高。
    - `model_batch`（可选，默认 1）：batch 大小。
    - `postnum`（可选，默认 1）：后处理线程数。
    - `frames_per_second`（可选，默认 1000）：输入线程节流上限。
    - `frame_decimation`（可选，默认 0）：每处理 1 帧后跳过 N 帧，`0` 表示不跳帧，可被 `io_info` 覆盖。
    - `track_config`（可选，模型级默认值）：
      - `enable_tracking`：是否启用跟踪（默认 true）。
      - `track_model_path`：跟踪 `.om` 模型路径。
      - `tracking_config`：跟踪阈值配置，对应 Tracking 的 setter：
        - `confidence_active_threshold`
        - `confidence_redetect_threshold`
        - `max_track_loss_frames`
        - `score_decay_factor`
    - `io_info[]`：每路输入/输出通道。
      - `input_path`：来源（如 `rtsp://...` 或文件）。
      - `input_type`：来源类型（如 `rtsp`）。
      - `output_path`：输出目的地；RTSP 时作为推流基址。
      - `output_type`：输出类型；`rtsp` 会启用推流线程。
      - `channel_id`：通道唯一 ID。
      - `frame_decimation`（可选）：覆盖模型级跳帧。
      - `rtsp_config`（可选，推流）：
        - `output_width` / `output_height`：编码尺寸，默认取模型输入尺寸。
        - `output_fps`：1–60，越界会回退到 25。
        - `transport`：`udp` 或 `tcp`。
        - `buffer_size`：socket buffer，单位字节。
        - `max_delay`：RTP 最大延迟，微秒。
      - `h264_config`（可选，编码）：
        - `gop_size`：1–300，越界回退到 16。
        - `rc_mode`：0=CBR，1=VBR，2=AVBR（>2 会被回退到 2）。
        - `max_bitrate`：单位 kbps，500–50000，越界回退到 10000。
        - `profile`：`baseline` | `main` | `high`。
      - 当模型级未提供 `track_config` 时，`enable_tracking` / `track_model_path` / `tracking_config` 可在通道级提供，含义相同。

### 如何调整 rtsp_config / h264_config
- 分辨率与帧率：`output_width` / `output_height` 应与业务需要和带宽匹配；`output_fps` 过高会增压编码与带宽，先用 15–25fps 验证再升高。
 - 分辨率与帧率：`output_width` / `output_height` 应与业务需要和带宽匹配；`output_fps` 过高会增压编码与带宽，先用 15–25fps 验证再升高。
- 传输方式：`transport` 选 `udp` 更低时延但可能丢包；`tcp` 更稳但时延高。
- 缓冲与延迟：弱网或远端链路可适当增大 `buffer_size`，`max_delay` 影响 RTP 抖动缓冲，过大时延上升，过小易卡顿。
- GOP 与码率：
  - `gop_size` 控制关键帧间隔（单位帧），直播常用 1–2 秒对应的帧数（如 25fps 用 25–50）。
  - `rc_mode`：`0` 恒定码率方便带宽预算；`1` 可变码率画质更高但带宽波动；`2` AVBR 折中。
  - `max_bitrate` 直接影响画质与带宽，按上行网络预留 20–30% 裕量设置。
- Profile：`baseline` 兼容性最好，`main` 常用，`high` 画质好但要求解码端支持。

**分辨率与码率/帧率参考（起步值，可按网络和画质再调）：**

| 分辨率 | 建议 fps | 建议 `max_bitrate` (kbps) | 建议 `gop_size` (对应 1–2s) |
| ------ | -------- | ------------------------- | --------------------------- |
| 1280x720 | 15–20 | 3000–5000 | 15–40 |
| 1920x1080 | 15–25 | 5000–8000 | 20–50 |
| 2560x1440 | 15–25 | 8000–12000 | 25–50 |
| 3840x2160 | 10–20 | 12000–20000 | 20–40 |

- 若链路较差或延迟敏感，优先降低 fps 或分辨率，再调低 `max_bitrate`。
- 关键帧密度越高（更小的 `gop_size`），seek/恢复更快但码率开销稍升。

### RTSP 最小示例
```json
{
  "device_config": [
    {
      "device_id": 0,
      "model_config": [
        {
          "model_path": "./model/yolov11n_110_rgb_640_raw_v0_bs1.om",
          "model_width": 640,
          "model_heigth": 640,
          "model_batch": 1,
          "frame_decimation": 5,
          "track_config": {
            "enable_tracking": true,
            "track_model_path": "./model/mixformerv2_online_small_bs1.om",
            "tracking_config": {
              "confidence_active_threshold": 0.6,
              "confidence_redetect_threshold": 0.6,
              "max_track_loss_frames": 5,
              "score_decay_factor": 0.98
            }
          },
          "io_info": [
            {
              "input_path": "rtsp://<src>",
              "input_type": "rtsp",
              "output_path": "rtsp://<sink>/uav",
              "output_type": "rtsp",
              "channel_id": 0,
              "rtsp_config": {
                "output_width": 1920,
                "output_height": 1080,
                "output_fps": 20,
                "transport": "udp",
                "buffer_size": 1024000,
                "max_delay": 200000
              },
              "h264_config": {
                "gop_size": 25,
                "rc_mode": 2,
                "max_bitrate": 10000,
                "profile": "main"
              }
            }
          ]
        }
      ]
    }
  ]
}
```

## 小贴士
- 相对路径从当前工作目录解析（通常是 `build/`）。
- `frame_decimation` 在处理完 1 帧后生效，例如 `5` 表示保留 1 帧、跳过后续 5 帧。
- 若在模型级关闭跟踪（`track_config.enable_tracking: false`），请确认通道级不会重新打开。
- 上真实流前，可用本地文件或自建 RTSP 服务先验证 JSON 配置。
