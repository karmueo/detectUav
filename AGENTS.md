# 仓库指南

## 项目结构与模块组织
- `src/`：Ascend ACL 主程序，含输入/输出、预处理、推理、后处理、跟踪、RTSP 推流，以及入口 `main.cpp`；可执行文件输出到 `build/src/out/`，调试符号保存在相同目录。
- `common/`：ACL Lite 公共封装（线程、日志、资源管理）；`inc/` 放跨模块头文件。
- `scripts/`：配置示例（`test.json`）、RTSP/H.264 辅助脚本、模型转换脚本（`atc.sh`、`atc_mixformer.sh`）、同步工具。
- `model/`：ONNX 源模型与 AIPP 配置；转换生成的 `.om` 也放此目录，并与 JSON 中的路径一致。
- `data/`：示例输入/输出与测试素材；避免提交体积过大或含敏感信息的资产。
- `build/`：CMake/Ninja 生成的编译产物与 `compile_commands.json`，不手工修改。
- `cmake/`：交叉编译与 Ascend 工具链相关的工具链文件；如需自定义，请复制后按实际路径调整。

## 构建、测试与开发命令
- 前置：已安装 Ascend 驱动/工具链，并配置 `SYSROOT`、`DDK_PATH`、`NPU_HOST_LIB`（未设默认 `/usr/local/Ascend/ascend-toolkit/latest`），需要 FFmpeg/OpenCV 及 live555（若开启 RTSP 推流）。
- 配置与编译（推荐 Ninja）：  
  `mkdir -p build && cd build && cmake -G Ninja .. && cmake --build . --target main test_mixformerv2_om`
- 运行主流程（在 `build/` 下）：  
  `./src/out/main ../scripts/test.json`（JSON 定义设备、模型路径、IO、跳帧、跟踪配置）。
- 跟踪回归测试（需 `init_img.jpg`、`track_img.jpg`、`ini_box.txt` 以及 `model/mixformerv2_online_small_bs1.om`）：  
  `./src/out/test_mixformerv2_om`
- 模型转换示例（在仓库根目录，Ascend 工具已入 PATH）：  
  `bash scripts/atc.sh` 或 `bash scripts/atc_mixformer.sh` 生成 `.om` 至 `model/`。
- 调试：`scripts/run_gdbserve.sh` 可用于远程调试；`scripts/sync_sysroot.sh` 帮助同步交叉编译环境。
- 清理与重编译：`cmake --build . --target clean && cmake --build .`，必要时删除 `build/` 重新配置。

## 代码风格与命名约定
- 使用 C++11，4 空格缩进，Allman 花括号；类用 PascalCase，函数 CamelCase，局部变量 lower_snake_case，常量多用 `k前缀`。
- 优先使用 `ACLLITE_LOG_*` 日志与资源封装；能用 RAII 便避免手工释放。
- 新模块放入 `src/` 并在 `src/CMakeLists.txt` 注册；公共头文件置于 `inc/` 或 `common/include/`。
- 统一格式：运行 `clang-format -i <file>`（配置见仓库 `.clang-format`，列宽 80）；提交前确保无多余警告。

## 测试指南
- 优先编写小而确定的测试，覆盖预处理/后处理计算、线程消息及跟踪 I/O；测试文件靠近对应模块存放，命名可用 `*_test.cpp`。
- 保持示例资产轻量并注明来源；勿在测试或配置中暴露凭据、私网地址或流密钥。
- 新功能提交前在 PR 中注明运行命令与关键日志，并在资产具备时确保 `test_mixformerv2_om` 通过。
- 若新增 JSON 配置字段，请补充脚本示例并描述缺省行为，避免运行时解析错误。
- 跟踪或视频类场景建议附一段性能或延时说明（样本帧率、batch、帧丢弃策略），便于复现。

## 提交与 PR 规范
- 保持现有提交习惯：简短、动作在前（如“优化…”“修复…”“重构…”），必要时注明模块。
- PR 请包含：变更摘要、配置/模型更新说明、测试证据（命令与日志片段）、已脱敏的 RTSP/模型路径；若有可视化输出变更再附截图。
- 若改动涉及线程或资源生命周期，请在描述中指出潜在风险点（如退出顺序、上下文复用），便于 reviewers 聚焦。
- 如关联需求/缺陷，请在标题或正文中注明 Issue/任务编号，保持双向可追溯。

## 安全与配置提示
- 不提交硬编码账户、RTSP 密钥或公网地址；将敏感信息放入本地私有 JSON 并在 PR 中用占位符示例。
- 环境变量 `LIVE555_INCLUDE_DIRS` 与 `LIVE555_LIBRARY_DIRS` 控制 RTSP 依赖路径；未配置时可关闭 `USE_LIVE555` 走 FFmpeg 路径。
- 模型与视频流路径建议使用相对路径，便于跨设备同步；大文件可通过外部存储同步，不直接入库。

## 开发约束

1. 在回答问题或修改代码前，如果有提高回答准确率和代码质量有帮助的信息你不确定时，可以问我补充完整。
2. 生成的函数定义时加上函数头中文注释。
3. 生成的变量声明时减少变量说明中文注释。
4. 生成工程代码时考虑合理的设计模式，做到模块化设计，松耦合，方便后续迭代。
