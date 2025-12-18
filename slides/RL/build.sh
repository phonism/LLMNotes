#!/bin/bash
# Slides 编译脚本

set -e

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${YELLOW}编译 Beamer Slides...${NC}"

cd "$(dirname "$0")"

# 编译两次确保目录正确
xelatex -shell-escape main.tex > /dev/null
xelatex -shell-escape main.tex > /dev/null

# 重命名输出
mv main.pdf RL_Slides.pdf

# 清理临时文件
echo -e "${YELLOW}清理临时文件...${NC}"
rm -f main.aux main.log main.nav main.out main.snm main.toc main.vrb
rm -rf _minted-main

echo -e "${GREEN}✓ 完成: RL_Slides.pdf${NC}"
