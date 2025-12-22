#!/bin/bash
# TikZ 图表编译脚本
# 从 Markdown 文件中提取 TikZ 源码并编译为 SVG
#
# 使用方法：
#   cd blogs/_figures && ./build.sh          # 编译所有图
#   cd blogs/_figures && ./build.sh name     # 只编译指定图

set -e
cd "$(dirname "$0")"

POSTS_DIR="../_posts"
OUTPUT_DIR="../assets/figures"
PREAMBLE="preamble.tex"
TMP_DIR=".tmp"
TARGET_NAME="$1"

# 颜色输出
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

log_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# 检查依赖
check_deps() {
  if ! command -v xelatex &> /dev/null; then
    log_error "xelatex 未安装"
    exit 1
  fi
  if command -v pdf2svg &> /dev/null; then
    PDF_TO_SVG="pdf2svg"
  elif command -v dvisvgm &> /dev/null; then
    PDF_TO_SVG="dvisvgm"
  else
    log_error "需要 pdf2svg 或 dvisvgm"
    exit 1
  fi
  log_info "使用 $PDF_TO_SVG 转换 SVG"
}

# 从临时文件编译图
compile_figure_from_file() {
  local name="$1"
  local content_file="$2"

  log_info "编译: $name"

  local tex_file="$TMP_DIR/$name.tex"

  # 合并 preamble + tikz code + end
  cat "$PREAMBLE" > "$tex_file"
  cat "$content_file" >> "$tex_file"
  echo '\end{document}' >> "$tex_file"

  # 编译
  pushd "$TMP_DIR" > /dev/null
  if xelatex -interaction=nonstopmode "$name.tex" > /dev/null 2>&1; then
    local svg_ok=0
    if [ "$PDF_TO_SVG" = "dvisvgm" ]; then
      dvisvgm --pdf "$name.pdf" -o "$name.svg" 2>/dev/null && svg_ok=1
    else
      pdf2svg "$name.pdf" "$name.svg" 2>/dev/null && svg_ok=1
    fi

    if [ "$svg_ok" -eq 1 ]; then
      mv "$name.svg" "../../assets/figures/"
      log_info "  -> $OUTPUT_DIR/$name.svg"
    else
      log_error "  SVG 转换失败: $name"
    fi
  else
    log_error "  xelatex 编译失败: $name"
    log_warn "  查看: _figures/$TMP_DIR/$name.log"
  fi
  popd > /dev/null
}

# 从单个 Markdown 文件提取 TikZ
extract_from_file() {
  local md_file="$1"
  local in_block=0
  local current_name=""
  local tmp_content=""

  mkdir -p "$TMP_DIR"
  tmp_content="$TMP_DIR/_content.tex"

  while IFS= read -r line || [ -n "$line" ]; do
    # 检测开始标记: <!-- tikz-source: name
    if [[ "$line" =~ ^'<!-- tikz-source: '([a-zA-Z0-9_-]+) ]]; then
      current_name="${BASH_REMATCH[1]}"
      in_block=1
      > "$tmp_content"  # 清空临时文件
      continue
    fi

    # 检测结束标记: -->
    if [ "$in_block" -eq 1 ] && [[ "$line" == "-->" ]]; then
      in_block=0
      # 如果指定了目标，只编译匹配的
      if [ -n "$TARGET_NAME" ] && [ "$current_name" != "$TARGET_NAME" ]; then
        continue
      fi
      compile_figure_from_file "$current_name" "$tmp_content"
      continue
    fi

    # 收集内容到临时文件
    if [ "$in_block" -eq 1 ]; then
      echo "$line" >> "$tmp_content"
    fi
  done < "$md_file"
}

# 清理临时文件
cleanup() {
  rm -rf "$TMP_DIR"
}

# 主流程
main() {
  check_deps
  mkdir -p "$OUTPUT_DIR"

  log_info "扫描 Markdown 文件..."

  local found=0
  # 递归扫描中文和英文目录
  while IFS= read -r -d '' md_file; do
    # 快速检查文件是否包含 tikz-source
    if grep -q "tikz-source:" "$md_file" 2>/dev/null; then
      found=1
      extract_from_file "$md_file"
    fi
  done < <(find "$POSTS_DIR" "../en/_posts" -name "*.md" -print0 2>/dev/null)

  if [ "$found" -eq 0 ]; then
    log_warn "未找到包含 tikz-source 的文件"
  fi

  cleanup
  log_info "完成!"
}

main
