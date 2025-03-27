# FunASR中文TTS字符错误率评估工具

## 简介

FunASRCERCalculator是一个专为中文语音识别评估设计的工具，它能够计算TTS（文本到语音）系统生成的语音与原始文本之间的字符错误率（CER）。该工具的特点是支持多音字处理，能够更准确地评估中文TTS系统的质量。

## 主要功能

- **多音字感知评估**：考虑中文多音字特性，避免因多音字导致的错误计算
- **详细错误分析**：提供字符级和词级错误分析，包括替换、删除、插入等错误类型
- **错误模式识别**：识别常见错误模式，如发音相似错误、声调错误等
- **批量评估**：支持批量处理多个音频文件
- **灵活的预处理选项**：可选择性忽略标点、空格和大小写

## 安装依赖

```bash
pip install funasr soundfile numpy pypinyin jieba python-Levenshtein
```

## 使用方法

### 基本用法

```python
from funasr_cer_calculator import FunASRCERCalculator
import soundfile as sf

# 初始化评估器
cer_calculator = FunASRCERCalculator(
    model_name="paraformer-zh",  # FunASR模型名称
    polyphone_path="polyphone.json",  # 多音字字典路径
    device="cuda:0"  # 推理设备
)

# 加载音频
audio_data, sample_rate = sf.read("example.wav")

# 计算CER
result = cer_calculator.compute_cer(
    audio=audio_data,
    sample_rate=sample_rate,
    reference_text="参考文本",
    consider_polyphones=True,  # 是否考虑多音字
    ignore_punctuation=True,  # 是否忽略标点
    ignore_space=True,  # 是否忽略空格
    lowercase=True  # 是否忽略大小写
)

print(f"CER: {result['cer']}")
print(f"WER: {result['wer']}")
```

### 详细分析

```python
# 获取详细分析结果
detailed_result = cer_calculator.compute_cer(
    audio=audio_data,
    sample_rate=sample_rate,
    reference_text="参考文本",
    consider_polyphones=True,
    detailed_analysis=True  # 启用详细分析
)

# 查看错误类型统计
error_counts = detailed_result['char_level']['error_counts']
print(f"替换错误: {error_counts.get('substitutions', 0)}")
print(f"删除错误: {error_counts.get('deletions', 0)}")
print(f"插入错误: {error_counts.get('insertions', 0)}")

# 查看常见错误模式
top_errors = detailed_result['error_patterns']['top_common_errors']
for err in top_errors[:5]:
    print(f"'{err['ref']}' → '{err['hyp']}' (出现 {err['count']} 次)")
```

### 批量处理

```python
# 批量计算CER
batch_results = cer_calculator.compute_batch_cer(
    audios=[audio1, audio2, audio3],
    sample_rates=[sr1, sr2, sr3],
    texts=["文本1", "文本2", "文本3"]
)

# 按文本特征分析CER
feature_analysis = cer_calculator.analyze_batch_by_text_features(
    audios=[audio1, audio2, audio3],
    sample_rates=[sr1, sr2, sr3],
    texts=["文本1", "文本2", "文本3"]
)

# 查看不同长度文本的CER
print(f"短文本CER: {feature_analysis['by_length']['short (<=10)']}")
print(f"中等长度文本CER: {feature_analysis['by_length']['medium (11-30)']}")
print(f"长文本CER: {feature_analysis['by_length']['long (>30)']}")
```

## 多音字处理

该工具通过以下方式处理多音字：

1. 加载多音字字典，包含多音字及其可能的读音
2. 在计算CER时，如果识别结果与参考文本在字符上不同，但拼音相同且是有效的多音字变体，则不计入错误

多音字字典格式示例：

```json
[
  {
    "char": "行",
    "pinyin": ["xing2", "hang2"]
  },
  {
    "char": "长",
    "pinyin": ["chang2", "zhang3"]
  }
]
```

## 高级功能

- **声调错误分析**：识别拼音相同但声调不同的错误
- **连续错误检测**：识别文本中连续出现的错误
- **发音相似错误分析**：识别因发音相似导致的错误

## 注意事项

- 确保FunASR模型已正确安装并可访问
- 多音字字典对于准确评估中文TTS系统至关重要
- 音频采样率建议为16kHz，以获得最佳ASR效果

## 示例应用场景

- TTS系统质量评估
- 语音识别系统性能测试
- 多音字处理算法研究
- 语音合成错误模式分析
