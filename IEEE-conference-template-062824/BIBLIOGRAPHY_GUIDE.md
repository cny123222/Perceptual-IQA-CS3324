# 📚 参考文献使用指南 - SMART-IQA 论文

## ✅ 系统已配置完成

你的论文现在使用 **BibTeX** 管理参考文献，配置如下：

```
IEEE-conference-template-062824/
├── IEEE-conference-template-062824.tex  (主论文文件)
├── references.bib                        (参考文献库 - 你添加文献到这里)
└── IEEEtran.bst                         (IEEE格式文件，已存在)
```

---

## 📝 如何添加参考文献

### 步骤1: 在 `references.bib` 中添加文献

打开 `references.bib`，在文件中添加 BibTeX 格式的条目：

#### 格式1：会议论文 (Conference Paper)

```bibtex
@inproceedings{作者姓_年份_简称,
  title={论文标题},
  author={作者1 and 作者2 and 作者3},
  booktitle={会议全称},
  pages={页码},
  year={年份}
}
```

**示例：**
```bibtex
@inproceedings{su2020hyperiq,
  title={Blindly assess image quality in the wild guided by a self-adaptive hyper network},
  author={Su, Shaolin and Yan, Qingsen and Zhu, Yu and Zhang, Cheng and Ge, Xin and Sun, Jinqiu and Zhang, Yanning},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={3667--3676},
  year={2020}
}
```

#### 格式2：期刊论文 (Journal Paper)

```bibtex
@article{作者姓_年份_简称,
  title={论文标题},
  author={作者1 and 作者2},
  journal={期刊名称},
  volume={卷号},
  number={期号},
  pages={页码},
  year={年份},
  publisher={出版社}
}
```

**示例：**
```bibtex
@article{hosu2020koniq,
  title={KonIQ-10k: An ecologically valid database for deep learning of blind image quality assessment},
  author={Hosu, Vlad and Lin, Hanhe and Sziranyi, Tamas and Saupe, Dietmar},
  journal={IEEE Transactions on Image Processing},
  volume={29},
  pages={4041--4056},
  year={2020},
  publisher={IEEE}
}
```

#### 格式3：arXiv 预印本

```bibtex
@article{作者姓_年份_简称,
  title={论文标题},
  author={作者1 and 作者2},
  journal={arXiv preprint arXiv:编号},
  year={年份}
}
```

**示例：**
```bibtex
@article{wu2023qalign,
  title={Q-Align: Teaching LMMs for visual scoring via discrete text-defined levels},
  author={Wu, Haoning and Zhang, Zicheng and Zhang, Weixia and Chen, Chaofeng and Li, Chunyi and Liao, Liang and Wang, Annan and Zhang, Erli and Sun, Wenxiu and Yan, Qiong and others},
  journal={arXiv preprint arXiv:2312.17090},
  year={2023}
}
```

---

### 步骤2: 在论文正文中引用

使用 `\cite{key}` 命令引用文献：

```latex
HyperIQA \cite{su2020hyperiq} proposed a self-adaptive hyper network...

Recent transformer-based methods \cite{liu2021swin,ke2021musiq,yang2022maniqa} have shown...

We evaluate our method on KonIQ-10k \cite{hosu2020koniq}.
```

**引用效果：**
- `\cite{su2020hyperiq}` → [1]
- `\cite{liu2021swin,ke2021musiq}` → [2, 3]

---

### 步骤3: 编译论文

**必须按照以下顺序编译：**

```bash
cd /root/Perceptual-IQA-CS3324/IEEE-conference-template-062824

pdflatex IEEE-conference-template-062824.tex
bibtex IEEE-conference-template-062824
pdflatex IEEE-conference-template-062824.tex
pdflatex IEEE-conference-template-062824.tex
```

**为什么要运行4次？**
1. `pdflatex` (第1次) - 生成 `.aux` 文件，记录引用信息
2. `bibtex` - 从 `references.bib` 中提取被引用的文献
3. `pdflatex` (第2次) - 将文献列表插入文档
4. `pdflatex` (第3次) - 更新交叉引用编号

---

## 🔍 如何从其他地方获取 BibTeX

### 方法1: Google Scholar

1. 在 Google Scholar 搜索论文
2. 点击论文标题下方的 **"引用"** 按钮
3. 选择 **BibTeX** 格式
4. 复制粘贴到 `references.bib`

### 方法2: arXiv

1. 打开论文页面（例如 https://arxiv.org/abs/2312.17090）
2. 右侧找到 **"Export BibTeX Citation"**
3. 复制粘贴到 `references.bib`

### 方法3: IEEE Xplore

1. 打开论文页面
2. 点击 **"Cite This"** 按钮
3. 选择 **BibTeX** 格式
4. 复制粘贴到 `references.bib`

### 方法4: ACM Digital Library

1. 打开论文页面
2. 点击 **"Export Citation"** 按钮
3. 选择 **BibTeX** 格式
4. 复制粘贴到 `references.bib`

### 方法5: DBLP (计算机科学论文专用)

1. 访问 https://dblp.org/
2. 搜索论文
3. 点击论文条目后的 **"export"** 按钮
4. 选择 **BibTeX**
5. 复制粘贴到 `references.bib`

---

## 📋 已预置的参考文献

我已经在 `references.bib` 中预置了以下文献：

### IQA 核心论文
- `su2020hyperiq` - HyperIQA (CVPR 2020)
- `liu2021swin` - Swin Transformer (ICCV 2021)

### 数据集
- `hosu2020koniq` - KonIQ-10k
- `fang2020perceptual` - SPAQ
- `lin2019kadid` - KADID-10K
- `li2023agiqa` - AGIQA-3K

### SOTA 方法
- `talebi2018nima` - NIMA (2018)
- `ying2020paq2piq` - PaQ-2-PiQ (2020)
- `ke2021musiq` - MUSIQ (2021)
- `golestaneh2022tres` - TReS (2022)
- `yang2022maniqa` - MANIQA (2022)
- `zhang2023liqe` - LIQE (2023)
- `wu2023qalign` - Q-Align (2023)

### Transformer 基础论文
- `vaswani2017attention` - Attention is All You Need
- `dosovitskiy2021vit` - Vision Transformer (ViT)

### 注意力机制
- `hu2018senet` - SENet
- `woo2018cbam` - CBAM

---

## 💡 常用引用示例

### Introduction 部分
```latex
Blind image quality assessment (BIQA) aims to predict perceptual quality 
without reference images \cite{talebi2018nima,su2020hyperiq}. Recent 
transformer-based approaches \cite{liu2021swin,ke2021musiq,yang2022maniqa} 
have shown promising results...
```

### Related Work 部分
```latex
\subsection{Transformer-based IQA Methods}
MUSIQ \cite{ke2021musiq} introduced multi-scale transformers for IQA...
MANIQA \cite{yang2022maniqa} leveraged multi-dimensional attention...
```

### Method 部分
```latex
We adopt Swin Transformer \cite{liu2021swin} as our feature extractor 
due to its hierarchical architecture and efficiency. Following HyperIQA 
\cite{su2020hyperiq}, we employ a hyper network to generate weights...
```

### Experiments 部分
```latex
We evaluate our method on KonIQ-10k \cite{hosu2020koniq}, 
SPAQ \cite{fang2020perceptual}, KADID-10K \cite{lin2019kadid}, 
and AGIQA-3K \cite{li2023agiqa}.
```

---

## ⚠️ 常见问题

### Q1: 编译后显示 `[?]` 而不是引用编号？
**A:** 你需要运行完整的编译流程（4次命令）。

### Q2: 参考文献列表是空的？
**A:** 确保你在正文中至少使用了一次 `\cite{key}`，BibTeX只会列出被引用的文献。

### Q3: BibTeX key 怎么命名？
**A:** 建议格式：`作者姓_年份_简称`  
- 例如：`su2020hyperiq`, `liu2021swin`, `yang2022maniqa`

### Q4: 我找不到某篇论文的 BibTeX？
**A:** 告诉我论文标题，我帮你查找并生成 BibTeX 条目。

### Q5: 如何引用多篇文献？
**A:** 用逗号分隔：`\cite{paper1,paper2,paper3}`

### Q6: 如何在文中多次引用同一篇文献？
**A:** 直接重复使用 `\cite{key}` 即可，编号会自动一致。

---

## 🎯 下一步

### 你现在需要做的：

1. **收集你要引用的论文**
   - 找到论文的 BibTeX 格式
   - 从 Google Scholar、arXiv、IEEE Xplore 等获取

2. **添加到 `references.bib`**
   - 打开 `references.bib` 文件
   - 粘贴 BibTeX 条目
   - 给每个条目一个清晰的 key

3. **在论文中引用**
   - 在 `.tex` 文件中使用 `\cite{key}`
   - 引用你添加的文献

4. **编译查看效果**
   - 运行完整的编译流程（4次命令）
   - 检查参考文献列表是否正确

---

## 📌 快速参考

| 命令 | 说明 |
|------|------|
| `\cite{key}` | 引用文献 |
| `\cite{key1,key2}` | 引用多篇文献 |
| `@inproceedings{}` | 会议论文 |
| `@article{}` | 期刊论文/arXiv |
| `@book{}` | 书籍 |
| `@misc{}` | 其他类型 |

---

**如果你有任何问题，直接告诉我：**
- "帮我查找XXX论文的BibTeX"
- "我这个BibTeX格式对吗？"
- "编译报错了，怎么办？"
- "我想引用一个网页/代码库，怎么写？"

我会立即帮你解决！ 🚀

