import re

with open('IEEE-conference-template-062824.tex', 'r', encoding='utf-8') as f:
    lines = f.readlines()

sections = []
current_section = None
current_line = 0

for i, line in enumerate(lines, 1):
    # Match section headers
    if re.search(r'\\section\*?\{', line):
        if current_section:
            sections.append((current_section, current_line, i-1))
        match = re.search(r'\\section\*?\{([^}]+)\}', line)
        current_section = match.group(1) if match else "Unknown"
        current_line = i
    elif re.search(r'\\appendix', line):
        if current_section:
            sections.append((current_section, current_line, i-1))
        current_section = "APPENDIX START"
        current_line = i
    elif re.search(r'\\bibliography', line):
        if current_section:
            sections.append((current_section, current_line, i-1))
        current_section = "Bibliography"
        current_line = i

# Add last section
if current_section:
    sections.append((current_section, current_line, len(lines)))

# Print results
print("=" * 80)
print("📊 论文结构分析 (Line Count)")
print("=" * 80)

appendix_started = False
main_text_lines = 0
appendix_lines = 0

for section, start, end in sections:
    line_count = end - start + 1
    
    if section == "APPENDIX START":
        appendix_started = True
        print("\n" + "=" * 80)
        print("📎 APPENDIX 开始")
        print("=" * 80)
        continue
    
    if not appendix_started and section != "Bibliography":
        main_text_lines += line_count
    elif appendix_started and section != "Bibliography":
        appendix_lines += line_count
    
    indent = "  " if appendix_started else ""
    print(f"{indent}{section:50s} | Lines {start:4d}-{end:4d} ({line_count:3d} lines)")

print("\n" + "=" * 80)
print(f"📝 正文总行数: {main_text_lines} lines")
print(f"📎 附录总行数: {appendix_lines} lines")
print(f"📄 文档总行数: {len(lines)} lines")
print("=" * 80)
