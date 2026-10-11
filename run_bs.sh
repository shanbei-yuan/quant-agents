#!/bin/bash
# 6 分片并行下载个股历史 (Baostock)
cd /mnt/d/quant-agents || exit 1
PY=/root/miniconda3/envs/crisp/bin/python
rm -f logs/bs.done logs/bs_*.log
for i in 0 1 2 3 4 5; do
  setsid $PY -u prefetch_bs_par.py 300 $i 6 2015-01-01 > logs/bs_$i.log 2>&1 < /dev/null &
  sleep 1
done
wait
echo "ALL_DONE $(date +%H:%M:%S)" > logs/bs.done
