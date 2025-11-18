#!/bin/bash

# 定义变量
SOURCE_FILE="./build/src/out/main"
SOURCE_SCRIPT="./scripts/atc.sh"
SOURCE_GDB="./scripts/run_gdbserve.sh"
SOURCE_ACL="./src/acl.json"
SOURCE_TEST="./scripts/test.json"
SOURCE_DATA="./data"
SOURCE_MODEL="./model"
REMOTE_USER="root"
REMOTE_HOST="192.168.1.111"
REMOTE_PATH="/root/work/yolov11_cann_video"

# 检查源文件是否存在
if [ ! -f "$SOURCE_FILE" ]; then
    echo "错误: 源文件 $SOURCE_FILE 不存在"
    exit 1
fi

if [ ! -f "$SOURCE_SCRIPT" ]; then
    echo "错误: 源脚本 $SOURCE_SCRIPT 不存在"
    exit 1
fi

if [ ! -f "$SOURCE_GDB" ]; then
    echo "错误: 源脚本 $SOURCE_GDB 不存在"
    exit 1
fi

if [ ! -f "$SOURCE_ACL" ]; then
    echo "错误: 源文件 $SOURCE_ACL 不存在"
    exit 1
fi

if [ ! -f "$SOURCE_TEST" ]; then
    echo "错误: 源文件 $SOURCE_TEST 不存在"
    exit 1
fi

if [ ! -d "$SOURCE_DATA" ]; then
    echo "错误: 源目录 $SOURCE_DATA 不存在"
    exit 1
fi

if [ ! -d "$SOURCE_MODEL" ]; then
    echo "错误: 源目录 $SOURCE_MODEL 不存在"
    exit 1
fi

# 同步文件到远程服务器
echo "正在同步文件到 $REMOTE_USER@$REMOTE_HOST:$REMOTE_PATH"
rsync -avz "$SOURCE_FILE" "$REMOTE_USER@$REMOTE_HOST:$REMOTE_PATH"
rsync -avz "$SOURCE_SCRIPT" "$REMOTE_USER@$REMOTE_HOST:$REMOTE_PATH"
rsync -avz "$SOURCE_GDB" "$REMOTE_USER@$REMOTE_HOST:$REMOTE_PATH"
rsync -avz "$SOURCE_ACL" "$REMOTE_USER@$REMOTE_HOST:$REMOTE_PATH"
rsync -avz "$SOURCE_TEST" "$REMOTE_USER@$REMOTE_HOST:$REMOTE_PATH"
rsync -avz "$SOURCE_DATA" "$REMOTE_USER@$REMOTE_HOST:$REMOTE_PATH"
rsync -avz "$SOURCE_MODEL" "$REMOTE_USER@$REMOTE_HOST:$REMOTE_PATH"

# 检查同步是否成功
if [ $? -eq 0 ]; then
    echo "✓ 文件同步成功"
else
    echo "✗ 文件同步失败"
    exit 1
fi