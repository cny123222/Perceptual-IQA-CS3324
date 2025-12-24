# 📊 Table 1 (SOTA对比表) 数据更新总结

**更新时间**: 2024-12-24  
**状态**: ✅ 已完成并编译成功

---

## ✅ **更新内容**

### **修正的数据**：

| Method | SRCC (旧) | SRCC (新) | PLCC (旧) | PLCC (新) | 状态 |
|--------|-----------|-----------|-----------|-----------|------|
| HyperIQA | 0.906 | 0.906 | 0.917 | 0.917 | ✅ 保持不变（原论文） |
| DBCNN | 0.884 | **0.875** | 0.968 | **0.884** | ✅ 已修正 |
| MUSIQ | 0.915 | **0.929** | 0.937 | **0.924** | ✅ 已修正 |
| UNIQUE | 0.893 | **0.896** | 0.900 | **0.901** | ✅ 已修正 |
| LIQE | 0.919 | **0.930** | 0.908 | **0.931** | ✅ 已修正 |
| StairIQA | 0.921 | 0.921 | 0.936 | 0.936 | ✅ 保持不变 |

### **新增方法**：

| Method | Backbone | SRCC | PLCC | 引用 |
|--------|---------|------|------|------|
| **WaDIQaM** | ResNet18 | 0.797 | 0.805 | bosse2017wadiqam |
| **PQR** | ResNet50 | 0.880 | 0.884 | zeng2021pqr |
| **SFA** | ResNet50 | 0.856 | 0.872 | li2022sfa |
| **CLIP-IQA+** | CLIP | 0.895 | 0.909 | wang2023clipiqa |

---

## 📊 **当前表格内容**

### **CNN-based Methods (7个)**：
1. NIMA: 0.558 / 0.590
2. WaDIQaM: 0.797 / 0.805 ⭐ 新增
3. PaQ-2-PiQ: 0.892 / 0.904
4. HyperIQA: 0.906 / 0.917
5. DBCNN: 0.875 / 0.884 ✏️ 已修正
6. PQR: 0.880 / 0.884 ⭐ 新增
7. SFA: 0.856 / 0.872 ⭐ 新增

### **Transformer-based Methods (7个)**：
1. TReS: 0.908 / 0.924
2. MANIQA: 0.920 / 0.930
3. StairIQA: 0.921 / 0.936
4. MUSIQ: 0.929 / 0.924 ✏️ 已修正
5. UNIQUE: 0.896 / 0.901 ✏️ 已修正
6. LIQE: 0.930 / 0.931 ✏️ 已修正
7. CLIP-IQA+: 0.895 / 0.909 ⭐ 新增

### **SMART-IQA (Ours) (3个)**：
1. Swin-Tiny: 0.9249 / 0.9360
2. Swin-Small: 0.9338 / 0.9455
3. **Swin-Base**: **0.9378 / 0.9485** ⭐ 最佳

---

## 📚 **参考文献更新**

### **新增的BibTeX条目**：

```bibtex
@inproceedings{bosse2017wadiqam, ...}     # WaDIQaM
@inproceedings{zeng2021pqr, ...}          # PQR
@article{li2022sfa, ...}                  # SFA
@inproceedings{wang2023clipiqa, ...}      # CLIP-IQA+
```

### **已有的引用**：
- ✅ talebi2018nima (NIMA)
- ✅ ying2020paq2piq (PaQ-2-PiQ)
- ✅ su2020hyperiq (HyperIQA)
- ✅ zhang2018dbcnn (DBCNN)
- ✅ golestaneh2022tres (TReS)
- ✅ yang2022maniqa (MANIQA)
- ✅ sun2024stairiqa (StairIQA)
- ✅ ke2021musiq (MUSIQ)
- ✅ zhang2021unique (UNIQUE)
- ✅ zhang2023liqe (LIQE)

---

## 🔍 **排名分析**

### **前5名（SRCC）**：
1. **SMART-IQA (Swin-Base)**: **0.9378** ⭐ 最佳
2. **SMART-IQA (Swin-Small)**: 0.9338
3. **LIQE**: 0.930
4. **MUSIQ**: 0.929
5. **SMART-IQA (Swin-Tiny)**: 0.9249

### **我们的优势**：
- **比LIQE高**: +0.0078 SRCC (+0.84%)
- **比MUSIQ高**: +0.0088 SRCC (+0.95%)
- **比StairIQA高**: +0.0168 SRCC (+1.82%)
- **比HyperIQA高**: +0.0318 SRCC (+3.51%)

---

## ✅ **编译状态**

```bash
cd IEEE-conference-template-062824/
pdflatex + bibtex + pdflatex + pdflatex

结果：✅ 成功
页数：8页
无错误：✅
无缺失引用：✅
```

---

## 📝 **文件位置**

### **主LaTeX文件**：
```
IEEE-conference-template-062824/IEEE-conference-template-062824.tex
```
- Table 1已更新（Line 101-128）

### **单独表格文件**：
```
IEEE-conference-template-062824/TABLE_1_SOTA_COMPARISON_UPDATED.tex
```
- 已同步更新

### **参考文献**：
```
IEEE-conference-template-062824/references.bib
```
- 新增4个引用

### **生成的PDF**：
```
IEEE-conference-template-062824/IEEE-conference-template-062824.pdf
```
- 8页，3.96 MB

---

## 🎯 **后续任务**

- [ ] 用户核对参考文献
- [ ] 确认所有数据准确
- [ ] 可选：添加更多baseline方法

---

## 📊 **表格预览**

```latex
Method               Backbone         SRCC    PLCC
--------------------------------------------------
CNN-based Methods:
  NIMA              InceptionNet     0.558   0.590
  WaDIQaM           ResNet18         0.797   0.805
  PaQ-2-PiQ         ResNet18         0.892   0.904
  HyperIQA          ResNet50         0.906   0.917
  DBCNN             ResNet50         0.875   0.884
  PQR               ResNet50         0.880   0.884
  SFA               ResNet50         0.856   0.872

Transformer-based:
  TReS              Transformer      0.908   0.924
  MANIQA            ViT-Small        0.920   0.930
  StairIQA          ResNet50         0.921   0.936
  MUSIQ             Multi-scale ViT  0.929   0.924
  UNIQUE            Swin-Tiny        0.896   0.901
  LIQE              MobileNet-Swin   0.930   0.931
  CLIP-IQA+         CLIP             0.895   0.909

SMART-IQA (Ours):
  Swin-Tiny         Swin-T (28M)     0.9249  0.9360
  Swin-Small        Swin-S (50M)     0.9338  0.9455
  Swin-Base         Swin-B (88M)     0.9378  0.9485 ⭐
```

---

**总结**: 所有数据已按照用户提供的准确数据更新，新增4个baseline方法，编译成功✅

