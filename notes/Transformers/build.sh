#!/bin/bash
# ==========================================
# LaTeX 编译脚本 (XeLaTeX)
# ==========================================

# 颜色输出
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
GRAY='\033[0;90m'
NC='\033[0m' # No Color

# 配置
MAIN_TEX="main.tex"
OUTPUT_DIR="output"
PDF_NAME="Transformers_Notes.pdf"
LOG_FILE="$OUTPUT_DIR/build.log"

# 错误处理：记录错误但不立即退出
error_exit() {
    echo -e "${RED}错误: $1${NC}"
    echo -e "${GRAY}查看日志: $LOG_FILE${NC}"
    exit 1
}

# 检查必要的工具
check_requirements() {
    echo -e "${YELLOW}[1/6] 检查编译环境...${NC}"

    if ! command -v xelatex &> /dev/null; then
        error_exit "未找到 xelatex，请安装 TeX Live 或 MacTeX"
    fi

    echo -e "${GREEN}✓ 编译环境检查通过${NC}"
}

# 创建输出目录
create_output_dir() {
    echo -e "${YELLOW}[2/6] 创建输出目录...${NC}"
    mkdir -p "$OUTPUT_DIR"
    # 清空或创建日志文件
    > "$LOG_FILE"
    echo -e "${GREEN}✓ 输出目录: $OUTPUT_DIR${NC}"
}

# 运行单次 xelatex 编译
run_xelatex() {
    local pass_name="$1"
    echo "  → $pass_name..."

    xelatex -interaction=nonstopmode \
            -output-directory="$OUTPUT_DIR" \
            "$MAIN_TEX" >> "$LOG_FILE" 2>&1

    local exit_code=$?

    # 检查是否生成了 PDF（比检查退出码更可靠）
    if [ ! -f "$OUTPUT_DIR/main.pdf" ]; then
        echo -e "${RED}  ✗ $pass_name 失败${NC}"
        # 显示最后几行错误
        echo -e "${GRAY}--- 错误信息 ---${NC}"
        tail -20 "$LOG_FILE" | grep -E "(Error|error|!)" | head -5
        return 1
    fi

    return 0
}

# 编译LaTeX文档
compile_latex() {
    echo -e "${YELLOW}[3/6] 编译LaTeX文档...${NC}"

    # 第一次编译 - 生成 aux 文件
    if ! run_xelatex "第一次编译 (生成引用信息)"; then
        error_exit "第一次编译失败"
    fi

    # 如果有参考文献，运行 bibtex
    if [ -f "$OUTPUT_DIR/main.aux" ] && grep -q "\\\\citation" "$OUTPUT_DIR/main.aux" 2>/dev/null; then
        echo "  → 处理参考文献 (bibtex)..."
        # bibtex 需要在输出目录运行，或指定路径
        cd "$OUTPUT_DIR"
        bibtex main >> build.log 2>&1 || true
        cd ..
    fi

    # 第二次编译 - 更新引用
    if ! run_xelatex "第二次编译 (更新引用)"; then
        error_exit "第二次编译失败"
    fi

    # 检查是否需要第三次编译（交叉引用变化）
    if grep -q "Rerun to get" "$LOG_FILE" 2>/dev/null; then
        if ! run_xelatex "第三次编译 (最终版本)"; then
            error_exit "第三次编译失败"
        fi
    else
        echo "  → 跳过第三次编译 (引用已稳定)"
    fi

    echo -e "${GREEN}✓ 编译完成${NC}"
}

# 复制PDF到根目录
copy_pdf() {
    echo -e "${YELLOW}[4/6] 复制PDF文件...${NC}"
    if [ -f "$OUTPUT_DIR/main.pdf" ]; then
        cp "$OUTPUT_DIR/main.pdf" "$PDF_NAME"
        echo -e "${GREEN}✓ PDF已保存到: $PDF_NAME${NC}"
    else
        error_exit "编译失败，未找到PDF文件"
    fi
}

# 显示统计信息
show_stats() {
    echo -e "${YELLOW}[5/6] 编译统计...${NC}"
    if [ -f "$PDF_NAME" ]; then
        SIZE=$(du -h "$PDF_NAME" | cut -f1)
        echo -e "${GREEN}✓ 文件大小: $SIZE${NC}"

        # 尝试获取页数
        if command -v pdfinfo &> /dev/null; then
            PAGES=$(pdfinfo "$PDF_NAME" 2>/dev/null | grep "Pages:" | awk '{print $2}')
            [ -n "$PAGES" ] && echo -e "${GREEN}✓ 页数: $PAGES${NC}"
        elif command -v mdls &> /dev/null; then
            # macOS 使用 mdls
            PAGES=$(mdls -name kMDItemNumberOfPages "$PDF_NAME" 2>/dev/null | awk '{print $3}')
            [ -n "$PAGES" ] && [ "$PAGES" != "(null)" ] && echo -e "${GREEN}✓ 页数: $PAGES${NC}"
        fi

        # 显示警告数量（如果有）
        if [ -f "$LOG_FILE" ]; then
            WARNINGS=$(grep -c "Warning" "$LOG_FILE" 2>/dev/null || echo "0")
            if [ "$WARNINGS" -gt 0 ]; then
                echo -e "${YELLOW}⚠ 警告数量: $WARNINGS${NC}"
            fi
        fi
    fi
}

# 显示帮助
show_help() {
    echo "用法: $0 [选项]"
    echo ""
    echo "选项:"
    echo "  -h, --help     显示帮助信息"
    echo "  -v, --verbose  显示详细编译输出"
    echo "  -c, --clean    编译前清理临时文件"
    echo ""
    echo "示例:"
    echo "  $0             # 正常编译"
    echo "  $0 -v          # 显示详细输出"
    echo "  $0 -c          # 清理后编译"
}

# 清理临时文件
clean_temp() {
    echo -e "${YELLOW}[6/6] 清理临时文件...${NC}"
    rm -rf "$OUTPUT_DIR" 2>/dev/null
    echo -e "${GREEN}✓ 清理完成${NC}"
}

# 主函数
main() {
    local verbose=false
    local clean=false

    # 解析参数
    while [[ $# -gt 0 ]]; do
        case $1 in
            -h|--help)
                show_help
                exit 0
                ;;
            -v|--verbose)
                verbose=true
                shift
                ;;
            -c|--clean)
                clean=true
                shift
                ;;
            *)
                echo -e "${RED}未知选项: $1${NC}"
                show_help
                exit 1
                ;;
        esac
    done

    echo -e "${GREEN}======================================${NC}"
    echo -e "${GREEN}  Transformers 笔记 - LaTeX 编译${NC}"
    echo -e "${GREEN}======================================${NC}"
    echo

    check_requirements
    create_output_dir

    [ "$clean" = true ] && clean_temp

    # 如果 verbose 模式，实时显示输出
    if [ "$verbose" = true ]; then
        LOG_FILE="/dev/stdout"
    fi

    compile_latex
    copy_pdf
    show_stats
    clean_temp

    echo
    echo -e "${GREEN}======================================${NC}"
    echo -e "${GREEN}  编译成功!${NC}"
    echo -e "${GREEN}======================================${NC}"
    echo -e "PDF文件: ${YELLOW}$PDF_NAME${NC}"
}

# 运行主函数
main "$@"
