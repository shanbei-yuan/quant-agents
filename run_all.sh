#!/bin/bash
# 一键流水线: 补齐缺失数据(优先东财一次请求拿全历史) -> 6 轮多智能体迭代训练
cd /mnt/d/quant-agents || exit 1
PY=/root/miniconda3/envs/crisp/bin/python
export MPLCONFIGDIR=/tmp/mpl

echo "=== [1/2] 补齐缺失数据 $(date +%H:%M:%S) ==="
$PY -u fill_missing.py > logs/fill.log 2>&1
echo "=== [1/2] 完成 $(date +%H:%M:%S) ==="

echo "=== [2/2] 开始训练 $(date +%H:%M:%S) ==="
$PY -u train_agents.py --universe 200 --rounds 6 --test-len 250 --cached-only > logs/train6.log 2>&1
echo "=== [2/2] 训练完成 $(date +%H:%M:%S) ==="

echo "ALL_DONE $(date +%F' '%H:%M:%S)" > logs/pipeline.done
