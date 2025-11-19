#!/bin/bash

# 同步脚本：从远程主机 root@192.168.1.111 同步指定目录到本机 SYSROOT

SYSROOT="/opt/ascend/sysroot"
REMOTE_HOST="root@192.168.1.111"

echo "Starting sync from $REMOTE_HOST to $SYSROOT"

# 同步 /usr/local/Ascend 到 $SYSROOT/usr/local/Ascend
echo "Syncing /usr/local/Ascend..."
rsync -a $REMOTE_HOST:/usr/local/Ascend $SYSROOT/usr/local/

# 同步 /usr/lib/ 到 $SYSROOT/usr/lib/
echo "Syncing /usr/lib/..."
rsync -a $REMOTE_HOST:/usr/lib/ $SYSROOT/usr/lib/

# 同步 /usr/include/ 到 $SYSROOT/usr/include/
echo "Syncing /usr/include/..."
rsync -a $REMOTE_HOST:/usr/include/ $SYSROOT/usr/include/

echo "Sync completed."