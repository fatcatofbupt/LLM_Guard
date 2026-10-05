#!/bin/bash
# 一键检查本地和远程模型同步状态

REMOTE="tts@116.169.217.30"
LOCAL_DIR="models"

echo "========== 同步状态检查 =========="
echo ""

# 获取本地和远程文件列表（排除 ._____temp 临时目录）
local_files=$(find "$LOCAL_DIR" -type f ! -path "*/._____temp/*" | sed 's|^models/||' | sort)
remote_files=$(ssh "$REMOTE" "find ~/models -type f ! -path '*/._____temp/*'" 2>/dev/null | sed 's|/home/tts/models/||' | sort)

# 计算文件数
local_count=$(echo "$local_files" | wc -l | tr -d ' ')
remote_count=$(echo "$remote_files" | wc -l | tr -d ' ')

echo "本地文件数: $local_count"
echo "远程文件数: $remote_count"
echo ""

# 对比差异
only_local=$(comm -23 <(echo "$local_files") <(echo "$remote_files"))
only_remote=$(comm -13 <(echo "$local_files") <(echo "$remote_files"))

if [ -n "$only_local" ]; then
    echo "❌ 以下文件只在本地，尚未同步到远程:"
    echo "$only_local" | sed 's/^/  - /'
    echo ""
fi

if [ -n "$only_remote" ]; then
    echo "⚠️  以下文件只在远程，本地没有:"
    echo "$only_remote" | sed 's/^/  - /'
    echo ""
fi

if [ -z "$only_local" ] && [ -z "$only_remote" ]; then
    echo "✅ 本地和远程文件完全一致，全部同步完成！"
else
    echo "⏳ 同步尚未完全完成。"
fi

echo ""
echo "========== 各模型大小对比 =========="
echo ""
printf "%-30s %-12s %-12s\n" "模型" "本地大小" "远程大小"
printf "%-30s %-12s %-12s\n" "------------------------------" "------------" "------------"

for model_dir in $(find "$LOCAL_DIR" -maxdepth 3 -type d | grep -v "._____temp" | grep -v "^$LOCAL_DIR$" | grep -v "^$LOCAL_DIR/[^/]*$") ; do
    model_name=$(echo "$model_dir" | sed "s|^$LOCAL_DIR/||")
    local_size=$(du -sh "$model_dir" 2>/dev/null | cut -f1)
    remote_size=$(ssh "$REMOTE" "du -sh ~/models/$model_name 2>/dev/null | cut -f1" 2>/dev/null)
    
    if [ "$local_size" = "$remote_size" ]; then
        status="✅"
    else
        status="❌"
    fi
    
    printf "%-30s %-12s %-12s %s\n" "$model_name" "$local_size" "$remote_size" "$status"
done

echo ""
echo "========== 自动监控状态 =========="
echo ""
if pgrep -f sync_monitor.sh > /dev/null 2>&1; then
    echo "🤖 自动监控脚本: 运行中"
else
    echo "🤖 自动监控脚本: 未运行"
fi

if [ -f sync_monitor.state ]; then
    echo "📋 监控状态文件:"
    cat sync_monitor.state | sed 's/^/  /'
fi

if [ -f sync_monitor.log ]; then
    echo ""
    echo "📜 最新监控日志:"
    tail -3 sync_monitor.log | sed 's/^/  /'
fi
