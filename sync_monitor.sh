#!/bin/bash
# 监控 models 目录下正在下载的模型，下载完成后自动 rsync 到远程服务器

REMOTE="tts@116.169.217.30"
LOCAL_DIR="/Users/duandingbo/Dev/LLM_Guard/models"
LOGFILE="/Users/duandingbo/Dev/LLM_Guard/sync_monitor.log"
STATEFILE="/Users/duandingbo/Dev/LLM_Guard/sync_monitor.state"

echo "$(date '+%Y-%m-%d %H:%M:%S') [INFO] 同步监控启动" >> "$LOGFILE"

# 初始化状态文件
if [ ! -f "$STATEFILE" ]; then
    touch "$STATEFILE"
fi

# 持续监控循环
while true; do
    # 1. 发现当前正在下载的模型（有 ._____temp 目录）
    find "$LOCAL_DIR" -maxdepth 3 -type d -name "._____temp" | while read temp_dir; do
        model_dir=$(dirname "$temp_dir")
        model_name=$(echo "$model_dir" | sed "s|^$LOCAL_DIR/||")
        
        # 如果状态文件中没有这个模型，标记为 downloading
        if ! grep -q "^$model_name:" "$STATEFILE" 2>/dev/null; then
            echo "$model_name:downloading" >> "$STATEFILE"
            temp_size=$(du -sh "$temp_dir" 2>/dev/null | cut -f1)
            echo "$(date '+%Y-%m-%d %H:%M:%S') [INFO] 发现新下载任务: $model_name (临时文件: $temp_size)" >> "$LOGFILE"
        fi
    done
    
    # 2. 检查之前标记为 downloading 的模型
    grep ":downloading$" "$STATEFILE" 2>/dev/null | while IFS=: read model_name status; do
        model_dir="$LOCAL_DIR/$model_name"
        temp_dir="$model_dir/._____temp"
        
        # 检查是否还有 modelscope 下载进程
        download_pid=$(ps aux | grep -E "modelscope.*$model_name" | grep -v grep | awk '{print $2}')
        
        if [ -z "$download_pid" ] && { [ ! -d "$temp_dir" ] || [ -z "$(ls -A "$temp_dir" 2>/dev/null)" ]; }; then
            # 下载进程结束且临时目录已清理，说明下载完成
            echo "$(date '+%Y-%m-%d %H:%M:%S') [INFO] $model_name 下载完成，开始 rsync 同步..." >> "$LOGFILE"
            
            # 执行 rsync
            rsync -avP --delete "$model_dir/" "$REMOTE:~/models/$model_name/" >> "$LOGFILE" 2>&1
            
            if [ $? -eq 0 ]; then
                echo "$(date '+%Y-%m-%d %H:%M:%S') [SUCCESS] $model_name 同步完成" >> "$LOGFILE"
                # 更新状态为 synced
                sed -i.bak "s|^$model_name:downloading|$model_name:synced|" "$STATEFILE" && rm -f "$STATEFILE.bak"
            else
                echo "$(date '+%Y-%m-%d %H:%M:%S') [ERROR] $model_name 同步失败，将在下次重试" >> "$LOGFILE"
            fi
        elif [ -n "$download_pid" ]; then
            # 仍在下载中，记录进度
            temp_size=$(du -sh "$temp_dir" 2>/dev/null | cut -f1)
            echo "$(date '+%Y-%m-%d %H:%M:%S') [PROGRESS] $model_name 下载中... 临时文件: $temp_size (PID: $download_pid)" >> "$LOGFILE"
        fi
    done
    
    # 每 30 秒检查一次
    sleep 30
done
