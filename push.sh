#!/bin/bash
# 一键推送: 先在 GitHub 网页新建空仓库 quant-agents (Public, 不要勾选 README/.gitignore/License),
# 然后运行本脚本。
cd /mnt/d/quant-agents || exit 1
git remote remove origin 2>/dev/null
git remote add origin git@github.com:shanbei-yuan/quant-agents.git
git branch -M main
echo "=== 即将推送 ==="
git log --oneline | head -3
git push -u origin main && echo "推送成功: https://github.com/shanbei-yuan/quant-agents"
