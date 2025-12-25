import re

with open('IEEE-conference-template-062824.tex', 'r', encoding='utf-8') as f:
    lines = f.readlines()

sections = []
current_section = None
current_subsection = None
current_line = 0
level = 0

for i, line in enumerate(lines, 1):
    # Match section headers
    if re.search(r'\\section\*?\{', line):
        match = re.search(r'\\section\*?\{([^}]+)\}', line)
        section_name = match.group(1) if match else "Unknown"
        sections.append((0, section_name, current_line, i-1))
        current_section = section_name
        current_line = i
        level = 0
    elif re.search(r'\\subsection\{', line):
        match = re.search(r'\\subsection\{([^}]+)\}', line)
        subsection_name = match.group(1) if match else "Unknown"
        sections.append((1, subsection_name, current_line, i-1))
        current_subsection = subsection_name
        current_line = i
        level = 1
    elif re.search(r'\\appendix', line):
        sections.append((0, "─── APPENDIX START ───", current_line, i-1))
        current_line = i
        level = -1

# Add last section
sections.append((level, "END", current_line, len(lines)))

# Print results
print("=" * 90)
print("📊 详细结构分析")
print("=" * 90)

in_appendix = False
main_sections = {}
appendix_sections = {}

for i, (lvl, name, start, end) in enumerate(sections):
    if name == "END":
        break
    
    if "APPENDIX START" in name:
        in_appendix = True
        print("\n" + "=" * 90)
        print("📎 附录部分")
        print("=" * 90)
        continue
    
    line_count = end - start + 1
    if line_count <= 0:
        continue
    
    indent = "  " if lvl == 1 else ""
    display_name = f"{indent}{name}"
    
    print(f"{display_name:55s} | {line_count:3d} lines (L{start:4d}-{end:4d})")
    
    # Collect stats
    if not in_appendix and lvl == 0:
        main_sections[name] = line_count
    elif in_appendix and lvl == 0:
        appendix_sections[name] = line_count

print("\n" + "=" * 90)
print("📈 统计摘要")
print("=" * 90)
print("\n正文各部分:")
for name, lines in main_sections.items():
    if "Bibliography" not in name and "Acknowledgment" not in name:
        print(f"  {name:40s}: {lines:3d} lines")

print("\n附录各部分:")
for name, lines in appendix_sections.items():
    print(f"  {name:55s}: {lines:3d} lines")

