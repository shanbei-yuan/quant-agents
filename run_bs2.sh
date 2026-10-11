#!/bin/bash
# 8 分片并行把股票池扩到 800 只(已缓存的会自动跳过)
cd /mnt/d/quant-agents || exit 1
PY=/root/miniconda3/envs/crisp/bin/python
N=8
rm -f logs/bs2.done logs/bs2_*.log
for i in $(seq 0 $((N-1))); do
  setsid $PY -u prefetch_bs_par.py 800 $i $N 2015-01-01 > logs/bs2_$i.log 2>&1 < /dev/null &
  sleep 1
done
wait
echo "ALL_DONE $(date +%H:%M:%S)" > logs/bs2.done
