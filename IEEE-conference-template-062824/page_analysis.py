# 基于section位置估算页码分布
# IEEE会议论文双栏格式，每页约50-60行文本+图表

# 从之前的分析得知：
# - 文档总共697行代码
# - 正文约390行代码（包含Introduction到Conclusion）
# - 附录约266行代码

# 考虑到图表、公式等占用额外空间，估算如下：

sections = [
    ("Title + Abstract + Keywords", 0, 36, 0.5),
    ("Introduction", 37, 48, 0.3),
    ("Related Work", 49, 73, 0.5),
    ("Method (含大图Figure 1)", 74, 206, 2.5),  # 包含architecture大图
    ("Experiments (含多个表格和图)", 207, 409, 5.0),  # 包含多个表格和图
    ("Conclusion", 410, 421, 0.3),
    ("References", 427, 429, 1.5),
    ("Appendix A", 432, 476, 1.0),
    ("Appendix B (含表格)", 477, 522, 1.0),
    ("Appendix C (含大表+图)", 523, 637, 2.5),
    ("Appendix D (含2张图)", 638, 697, 1.0),
]

print("=" * 90)
print("📄 页码分布估算 (基于实际PDF结构)")
print("=" * 90)
print()

cumulative = 0
main_pages = 0
appendix_pages = 0

for name, start, end, pages in sections:
    cumulative += pages
    if "Appendix" in name:
        appendix_pages += pages
        marker = "📎"
    else:
        main_pages += pages
        marker = "📝"
    
    print(f"{marker} {name:40s}: ~{pages:4.1f} pages (累计: {cumulative:5.1f})")

print()
print("=" * 90)
print(f"📝 正文总页数: ~{main_pages:.1f} pages")
print(f"📎 附录总页数: ~{appendix_pages:.1f} pages")
print(f"📄 总页数估算: ~{cumulative:.1f} pages (实际: 17 pages)")
print("=" * 90)

# 详细分析各部分占比
print()
print("=" * 90)
print("📊 各部分占比分析")
print("=" * 90)
print()
print("正文部分:")
for name, start, end, pages in sections[:7]:
    pct = (pages / 17) * 100
    bar = "█" * int(pct / 2)
    print(f"  {name:40s}: {pages:4.1f}p ({pct:5.1f}%) {bar}")

print()
print("附录部分:")
for name, start, end, pages in sections[7:]:
    pct = (pages / 17) * 100
    bar = "█" * int(pct / 2)
    print(f"  {name:40s}: {pages:4.1f}p ({pct:5.1f}%) {bar}")

