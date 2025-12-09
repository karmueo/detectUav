#!/bin/bash

# 定义变量
SOURCE_FILE="./build/src/out/main"
SOURCE_SCRIPTS="./scripts"
SOURCE_ACL="./src/acl.json"
SOURCE_DATA="./data"
SOURCE_MODEL="./model"
SOURCE_TEST_MIXFORMER="./build/src/out/test_mixformerv2_om"
SOURCE_TEST_HDMI_OUTPUT="./build/src/out/test_hdmi_output"
REMOTE_USER="root"
REMOTE_HOST="192.168.1.111"
REMOTE_PATH="/root/work/AntiUav"

# 检查源文件是否存在
if [ ! -f "$SOURCE_FILE" ]; then
    echo "错误: 源文件 $SOURCE_FILE 不存在"
    exit 1
fi

if [ ! -d "$SOURCE_SCRIPTS" ]; then
    echo "错误: 源目录 $SOURCE_SCRIPTS 不存在"
    exit 1
fi

if [ ! -f "$SOURCE_ACL" ]; then
    echo "错误: 源文件 $SOURCE_ACL 不存在"
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

if [ ! -f "$SOURCE_TEST_MIXFORMER" ]; then
    echo "错误: 源文件 $SOURCE_TEST_MIXFORMER 不存在"
    exit 1
fi

if [ ! -f "$SOURCE_TEST_HDMI_OUTPUT" ]; then
    echo "错误: 源文件 $SOURCE_TEST_HDMI_OUTPUT 不存在"
    exit 1
fi

# 同步文件到远程服务器
echo "正在同步文件到 $REMOTE_USER@$REMOTE_HOST:$REMOTE_PATH"
rsync -az "$SOURCE_FILE" "$REMOTE_USER@$REMOTE_HOST:$REMOTE_PATH"
rsync -az "$SOURCE_SCRIPTS" "$REMOTE_USER@$REMOTE_HOST:$REMOTE_PATH"
rsync -az "$SOURCE_ACL" "$REMOTE_USER@$REMOTE_HOST:$REMOTE_PATH"
rsync -az "$SOURCE_DATA" "$REMOTE_USER@$REMOTE_HOST:$REMOTE_PATH"
rsync -az "$SOURCE_MODEL" "$REMOTE_USER@$REMOTE_HOST:$REMOTE_PATH"
rsync -az "$SOURCE_TEST_MIXFORMER" "$REMOTE_USER@$REMOTE_HOST:$REMOTE_PATH"
rsync -az "$SOURCE_TEST_HDMI_OUTPUT" "$REMOTE_USER@$REMOTE_HOST:$REMOTE_PATH"

# 检查同步是否成功
if [ $? -eq 0 ]; then
    echo "✓ 文件同步成功"
else
    echo "✗ 文件同步失败"
    exit 1
fi