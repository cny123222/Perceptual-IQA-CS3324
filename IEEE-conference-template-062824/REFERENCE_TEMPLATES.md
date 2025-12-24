# 📋 BibTeX 参考文献模板速查表

## 🚀 快速开始

**你只需要做3件事：**

1. **复制下面的模板** → 粘贴到 `references.bib` 文件末尾
2. **填写信息** → 替换模板中的占位符
3. **在论文中引用** → 使用 `\cite{你的key}`

---

## 📚 常用模板

### 1️⃣ 会议论文 (Conference Paper)

**最常用！CVPR、ICCV、ECCV等都用这个**

```bibtex
@inproceedings{作者姓2024简称,
  title={论文完整标题},
  author={作者1 and 作者2 and 作者3},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={1234--5678},
  year={2024}
}
```

**示例：**
```bibtex
@inproceedings{zhang2024awesome,
  title={Awesome Image Quality Assessment with Deep Learning},
  author={Zhang, Wei and Li, Ming and Wang, Jian},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={12345--12354},
  year={2024}
}
```

**引用方式：** `\cite{zhang2024awesome}`

---

### 2️⃣ 期刊论文 (Journal Paper)

**IEEE TIP、TPAMI、IJCV等期刊**

```bibtex
@article{作者姓2024简称,
  title={论文完整标题},
  author={作者1 and 作者2},
  journal={IEEE Transactions on Image Processing},
  volume={33},
  number={5},
  pages={1234--5678},
  year={2024},
  publisher={IEEE}
}
```

**示例：**
```bibtex
@article{chen2024deep,
  title={Deep Learning for No-Reference Image Quality Assessment},
  author={Chen, Nuoyan and Liu, Xiaoming},
  journal={IEEE Transactions on Image Processing},
  volume={33},
  number={8},
  pages={4567--4580},
  year={2024},
  publisher={IEEE}
}
```

---

### 3️⃣ arXiv 预印本

**最新论文，还没正式发表**

```bibtex
@article{作者姓2024简称,
  title={论文完整标题},
  author={作者1 and 作者2},
  journal={arXiv preprint arXiv:2401.12345},
  year={2024}
}
```

**示例：**
```bibtex
@article{wang2024latest,
  title={Latest Advances in Vision Transformers for IQA},
  author={Wang, Hao and Zhang, Lei},
  journal={arXiv preprint arXiv:2401.12345},
  year={2024}
}
```

---

### 4️⃣ 书籍 (Book)

```bibtex
@book{作者姓2024,
  title={书名},
  author={作者},
  year={2024},
  publisher={出版社}
}
```

---

### 5️⃣ 书籍章节 (Book Chapter)

```bibtex
@incollection{作者姓2024简称,
  title={章节标题},
  author={作者1 and 作者2},
  booktitle={书名},
  pages={123--456},
  year={2024},
  publisher={出版社}
}
```

---

### 6️⃣ 技术报告 (Technical Report)

```bibtex
@techreport{作者姓2024简称,
  title={报告标题},
  author={作者},
  institution={机构名称},
  year={2024}
}
```

---

### 7️⃣ 硕士/博士论文

```bibtex
@phdthesis{作者姓2024,
  title={论文标题},
  author={作者},
  school={大学名称},
  year={2024}
}

@mastersthesis{作者姓2024,
  title={论文标题},
  author={作者},
  school={大学名称},
  year={2024}
}
```

---

### 8️⃣ 网页/在线资源

```bibtex
@misc{作者姓2024简称,
  title={网页标题},
  author={作者},
  year={2024},
  howpublished={\url{https://example.com}},
  note={Accessed: 2024-12-24}
}
```

---

## 🎯 实战示例：如何从Google Scholar获取BibTeX

### 步骤演示

假设你想引用 **HyperIQA** 论文：

1. **Google Scholar搜索** → "HyperIQA CVPR 2020"

2. **点击"引用"按钮** → 选择"BibTeX"

3. **复制得到的内容：**
```bibtex
@inproceedings{su2020blindly,
  title={Blindly assess image quality in the wild guided by a self-adaptive hyper network},
  author={Su, Shaolin and Yan, Qingsen and Zhu, Yu and Zhang, Cheng and Ge, Xin and Sun, Jinqiu and Zhang, Yanning},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={3667--3676},
  year={2020}
}
```

4. **粘贴到 `references.bib` 文件末尾**

5. **在论文中引用：**
```latex
HyperIQA \cite{su2020blindly} proposed a self-adaptive hyper network...
```

---

## 💡 命名规范建议

### Key的命名格式：`作者姓_年份_关键词`

| 示例 | 说明 |
|------|------|
| `su2020hyperiq` | 第一作者Su，2020年，HyperIQ |
| `liu2021swin` | 第一作者Liu，2021年，Swin |
| `chen2024deep` | 第一作者Chen，2024年，Deep |

**好处：**
- ✅ 一眼看出是谁的论文
- ✅ 一眼看出年份
- ✅ 不会重复

---

## 🔧 常见问题解决

### Q1: 作者太多怎么办？

**A:** 超过3个作者，后面用 `and others`：

```bibtex
author={Zhang, Wei and Li, Ming and Wang, Jian and others}
```

### Q2: 页码范围怎么写？

**A:** 使用 **双横线** `--`：

```bibtex
pages={1234--5678}    ✅ 正确
pages={1234-5678}     ❌ 错误
```

### Q3: 会议名称太长怎么办？

**A:** 可以适当缩写，保持一致即可：

```bibtex
booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition}
booktitle={Proceedings of the IEEE/CVF CVPR}  % 也可以
booktitle={CVPR}  % 最简洁
```

### Q4: 如何引用多篇文献？

**A:** 用逗号分隔：

```latex
Recent works \cite{su2020hyperiq,liu2021swin,yang2022maniqa} have shown...
```

### Q5: 编译后显示 [?] 怎么办？

**A:** 你需要运行**完整的4次编译**：

```bash
pdflatex paper.tex
bibtex paper
pdflatex paper.tex
pdflatex paper.tex
```

---

## 📝 你现在要做的

### 第1步：收集你要引用的论文

列出所有需要引用的论文清单：
- [ ] HyperIQA (已添加 ✅)
- [ ] Swin Transformer (已添加 ✅)
- [ ] KonIQ-10k数据集 (已添加 ✅)
- [ ] 你的其他论文...

### 第2步：获取BibTeX

对于每篇论文：
1. 在Google Scholar搜索
2. 点击"引用" → 选择"BibTeX"
3. 复制内容

### 第3步：添加到 references.bib

打开 `references.bib`，滚动到文件末尾，粘贴新的条目。

### 第4步：在论文中引用

在 `.tex` 文件中使用 `\cite{key}`。

### 第5步：编译查看效果

运行完整的编译流程。

---

## 🆘 需要帮助？

**直接告诉我：**

```
"帮我查找XXX论文的BibTeX"
"我想引用MUSIQ论文，怎么写？"
"这个BibTeX格式对吗？"
```

我会立即帮你生成！🚀

---

## 📌 已预置的参考文献

你的 `references.bib` 已经包含了以下文献，可以直接引用：

| Key | 论文 | 类型 |
|-----|------|------|
| `su2020hyperiq` | HyperIQA | IQA方法 |
| `liu2021swin` | Swin Transformer | Backbone |
| `hosu2020koniq` | KonIQ-10k | 数据集 |
| `fang2020perceptual` | SPAQ | 数据集 |
| `lin2019kadid` | KADID-10K | 数据集 |
| `li2023agiqa` | AGIQA-3K | 数据集 |
| `talebi2018nima` | NIMA | SOTA方法 |
| `ke2021musiq` | MUSIQ | SOTA方法 |
| `yang2022maniqa` | MANIQA | SOTA方法 |
| `zhang2023liqe` | LIQE | SOTA方法 |
| `wu2023qalign` | Q-Align | SOTA方法 |
| `vaswani2017attention` | Attention is All You Need | Transformer |
| `dosovitskiy2021vit` | Vision Transformer | ViT |
| `hu2018senet` | SENet | 注意力机制 |
| `woo2018cbam` | CBAM | 注意力机制 |

**使用示例：**
```latex
We adopt Swin Transformer \cite{liu2021swin} as our backbone...
We evaluate on KonIQ-10k \cite{hosu2020koniq} and SPAQ \cite{fang2020perceptual}...
```

---

**准备好了吗？开始添加你的参考文献吧！** 📚✨

