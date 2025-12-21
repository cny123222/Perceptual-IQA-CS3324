# 发现真正的速度问题！

## 关键发现

对比原始HyperIQA代码后，发现我们添加了**图像预加载和缓存机制**！

### 在 `folders.py` 的 `Koniq_10kFolder.__init__()` 中：

```python
# 我们添加的代码：
self._resized_cache = {}
print(f'Pre-loading {len(unique_paths)} unique images into cache...')
for path in tqdm(unique_paths, desc='  Loading images', unit='img'):
    img = pil_loader(path)
    self._resized_cache[path] = self.resize_transform(img)
```

### 问题分析

1. **原始HyperIQA**: 每次从磁盘读取图像 → 简单直接
2. **我们的版本**: 预加载所有图像到内存 → 看似优化，实际可能更慢

### 为什么会更慢？

可能的原因：
1. **缓存查找开销**: 每次都要查dict (`self._resized_cache.get(path)`)
2. **内存压力**: 7046张训练图全在内存中，可能导致swap
3. **CPU缓存失效**: 大量内存占用影响CPU缓存效率
4. **Python对象开销**: PIL Image对象在内存中的开销

### 解决方案

**移除图像预加载缓存机制，恢复原始的直接读取方式！**

原始HyperIQA的`__getitem__`:
```python
def __getitem__(self, index):
    path, target = self.samples[index]
    sample = pil_loader(path)  # 每次直接从磁盘读取
    sample = self.transform(sample)
    return sample, target
```

我们的版本:
```python
def __getitem__(self, index):
    path, target = self.samples[index]
    cached_img = self._resized_cache.get(path)  # 从缓存读取
    if cached_img:
        sample = self.crop_transform(cached_img)
    else:
        sample = pil_loader(path)
        ...
    return sample, target
```

## 行动计划

1. 移除folders.py中的所有缓存相关代码
2. 恢复原始的__getitem__实现
3. 重新测试训练速度
