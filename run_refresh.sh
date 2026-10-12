#!/bin/bash
# 8 分片并行增量刷新行情缓存（含指数）
cd /mnt/d/quant-agents || exit 1
PY=/root/miniconda3/envs/crisp/bin/python
N=8
rm -f logs/refresh.done logs/refresh_*.log
for i in $(seq 0 $((N-1))); do
  setsid $PY -u refresh_data.py $i $N 40 > logs/refresh_$i.log 2>&1 < /dev/null &
  sleep 1
done
wait
echo "REFRESH_DONE $(date +%F' '%H:%M:%S)" > logs/refresh.done
