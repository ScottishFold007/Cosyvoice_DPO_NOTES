# FunASRCERCalculator - 中文语音识别字符错误率评估工具

## 1. 工具概述

`FunASRCERCalculator` 是一个基于阿里达摩院 FunASR 语音识别模型的中文语音评估工具，专门用于计算和分析中文语音合成（TTS）系统的字符错误率（Character Error Rate, CER）。该工具通过将合成的语音转换为文本，并与参考文本进行比较，从而评估语音合成的准确性。

## 2. 主要功能

- **基础字符错误率计算**：计算语音与参考文本之间的字符错误率
- **拼音级别评估**：支持基于拼音的比较，有效解决多音字问题
- **详细错误分析**：提供字符级和词级的插入、删除、替换错误统计
- **批量处理能力**：支持批量音频的CER计算
- **文本特征分析**：按文本长度和内容类型分析CER表现
- **错误模式识别**：分析识别常见的错误模式，包括：
  - 相似发音字符混淆
  - 多音字错误
  - 连续错误
  - 声调错误

## 3. 技术实现

### 3.1 核心依赖

- **FunASR**：通过ModelScope加载的语音识别模型
- **Levenshtein**：用于计算编辑距离和错误分析
- **pypinyin**：用于中文转拼音处理
- **jieba**：用于中文分词
- **numpy**：用于数值计算
- **soundfile**：用于音频文件处理

### 3.2 工作流程

1. **初始化**：加载FunASR语音识别模型
2. **音频处理**：将输入的音频数据保存为临时文件
3. **语音识别**：使用FunASR模型将音频转换为文本
4. **文本预处理**：对参考文本和识别文本进行标准化处理
5. **错误率计算**：计算字符错误率或拼音错误率
6. **错误分析**：根据需要进行详细的错误模式分析

## 4. API详解

### 4.1 初始化

```python
calculator = FunASRCERCalculator(model_name="damo/speech_paraformer-large_asr_nat-zh-cn")
```

- **model_name**：指定使用的FunASR模型，默认使用达摩院的中文语音识别大模型

### 4.2 文本预处理

```python
processed_text = calculator.preprocess_text(text, remove_punctuation=True, convert_to_lower=True)
```

- **text**：输入文本
- **remove_punctuation**：是否移除标点符号
- **convert_to_lower**：是否转换为小写

### 4.3 计算字符错误率

```python
result = calculator.compute_cer(audio, sr, reference_text, use_pinyin=False, detailed_analysis=False)
```

- **audio**：音频数据（numpy数组）
- **sr**：采样率
- **reference_text**：参考文本
- **use_pinyin**：是否使用拼音进行比较
- **detailed_analysis**：是否返回详细错误分析

### 4.4 批量计算CER

```python
results = calculator.compute_batch_cer(audio_list, sample_rates, reference_texts)
```

- **audio_list**：音频数据列表
- **sample_rates**：采样率列表
- **reference_texts**：参考文本列表

### 4.5 按文本特征分析CER

```python
analysis = calculator.analyze_cer_by_text_features(audio_list, sample_rates, reference_texts)
```

返回按文本长度和内容类型分组的CER分析结果。

### 4.6 错误模式分析

```python
error_patterns = calculator.analyze_error_patterns(audio, sr, reference_text)
```

分析并返回详细的错误模式，包括相似发音错误、多音字错误、连续错误和声调错误。

## 5. 使用示例

### 5.1 基础CER计算

```python
import numpy as np
import soundfile as sf

# 加载音频文件
audio, sr = sf.read('test_audio.wav')

# 初始化计算器
calculator = FunASRCERCalculator()

# 计算CER
reference_text = "今天天气真不错"
cer = calculator.compute_cer(audio, sr, reference_text)
print(f"字符错误率: {cer:.2%}")
```

### 5.2 详细错误分析

```python
# 获取详细错误分析
detailed_result = calculator.compute_cer(audio, sr, reference_text, detailed_analysis=True)

# 打印字符级错误
print(f"插入错误: {detailed_result['char_level']['insertions']}")
print(f"删除错误: {detailed_result['char_level']['deletions']}")
print(f"替换错误: {detailed_result['char_level']['substitutions']}")

# 打印词级错误率
print(f"词错误率: {detailed_result['word_level']['word_error_rate']:.2%}")
```

### 5.3 错误模式分析

```python
# 分析错误模式
error_patterns = calculator.analyze_error_patterns(audio, sr, reference_text)

# 打印相似发音错误
for error in error_patterns['error_patterns']['similar_sound_errors']:
    print(f"位置 {error['position']}: 参考字符 '{error['ref_char']}' 被识别为 '{error['hyp_char']}', 拼音: {error['pinyin']}")

# 打印声调错误
for error in error_patterns['error_patterns']['tone_errors']:
    print(f"位置 {error['position']}: 参考拼音 '{error['ref_pinyin']}' 被识别为 '{error['hyp_pinyin']}'")
```

## 6. 应用场景

- **语音合成系统评估**：评估TTS系统的发音准确性
- **语音识别系统测试**：测试ASR系统的中文识别能力
- **多音字处理研究**：研究和改进多音字的处理方法
- **语音质量监控**：在生产环境中监控语音合成质量
- **错误模式分析**：分析并改进特定类型的语音合成错误

## 7. 注意事项

- 该工具需要网络连接以下载和使用ModelScope中的FunASR模型
- 处理大量音频时，请注意临时文件的存储空间
- 对于特殊领域的文本，可能需要自定义文本预处理逻辑
- `_find_polyphonic_errors`方法需要多音字词典才能完全实现，当前版本为简化实现

## 8. 未来改进方向

- 添加更多语音识别模型的支持
- 完善多音字错误检测功能
- 增加更多文本特征分析维度
- 提供可视化错误分析报告
- 优化批量处理性能
- 增加对方言和特殊领域文本的支持
