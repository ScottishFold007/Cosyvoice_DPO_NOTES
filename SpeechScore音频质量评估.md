

# 音频质量评估工具

基于 SpeechScore 封装的音频质量评估代码，支持单个或多个音频文件的评估，并将结果保存为 JSON 文件。

## 实现代码

```python:audio_quality_evaluator.py
#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import json
import argparse
from pathlib import Path
import pprint
from typing import List, Dict, Union, Optional
from speechscore import SpeechScore

class AudioQualityEvaluator:
    """音频质量评估工具，基于 SpeechScore 封装"""
    
    def __init__(self, metrics: Optional[List[str]] = None):
        """
        初始化评估器
        
        Args:
            metrics: 要评估的指标列表，如果为 None，则使用所有支持的指标
        """
        # 默认使用所有支持的指标
        if metrics is None:
            metrics = [
                'SRMR', 'PESQ', 'NB_PESQ', 'STOI', 'SISDR', 
                'FWSEGSNR', 'LSD', 'BSSEval', 'DNSMOS', 
                'SNR', 'SSNR', 'LLR', 'CSIG', 'CBAK', 
                'COVL', 'MCD'
            ]
        
        self.metrics = metrics
        self.speech_score = SpeechScore(metrics)
        
    def evaluate(self, 
                test_path: str, 
                reference_path: Optional[str] = None, 
                window: Optional[float] = None, 
                score_rate: int = 16000, 
                return_mean: bool = False) -> Dict:
        """
        评估音频质量
        
        Args:
            test_path: 测试音频文件或目录路径
            reference_path: 参考音频文件或目录路径，对于不需要参考的指标可以为 None
            window: 窗口大小（秒），None 表示处理整个音频
            score_rate: 采样率
            return_mean: 是否返回平均分数
            
        Returns:
            评估结果字典
        """
        # 检查是否只使用不需要参考的指标
        non_reference_metrics = ['DNSMOS', 'SRMR']
        only_non_reference = all(metric in non_reference_metrics for metric in self.metrics)
        
        # 如果只使用不需要参考的指标，reference_path 可以为 None
        if reference_path is None and not only_non_reference:
            print("警告: 部分指标需要参考音频，但未提供 reference_path。只有不需要参考的指标会被计算。")
            # 过滤掉需要参考的指标
            filtered_metrics = [m for m in self.metrics if m in non_reference_metrics]
            if not filtered_metrics:
                raise ValueError("没有可用的指标！请提供 reference_path 或选择不需要参考的指标。")
            temp_evaluator = SpeechScore(filtered_metrics)
            scores = temp_evaluator(test_path=test_path, reference_path=None, 
                                   window=window, score_rate=score_rate, return_mean=return_mean)
        else:
            # 正常评估
            scores = self.speech_score(test_path=test_path, reference_path=reference_path, 
                                      window=window, score_rate=score_rate, return_mean=return_mean)
        
        return scores
    
    def save_results(self, scores: Dict, output_path: str):
        """
        保存评估结果到 JSON 文件
        
        Args:
            scores: 评估结果字典
            output_path: 输出文件路径
        """
        # 确保输出目录存在
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        
        # 保存为 JSON 文件
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(scores, f, indent=4, ensure_ascii=False)
        
        print(f"评估结果已保存到: {output_path}")


def main():
    """命令行入口函数"""
    parser = argparse.ArgumentParser(description='音频质量评估工具')
    parser.add_argument('--test', required=True, help='测试音频文件或目录路径')
    parser.add_argument('--reference', help='参考音频文件或目录路径（对于某些指标是必需的）')
    parser.add_argument('--output', required=True, help='输出 JSON 文件路径')
    parser.add_argument('--metrics', nargs='+', help='要评估的指标列表，用空格分隔')
    parser.add_argument('--sample_rate', type=int, default=16000, help='采样率')
    parser.add_argument('--window', type=float, help='窗口大小（秒），不指定则处理整个音频')
    parser.add_argument('--mean', action='store_true', help='是否返回平均分数')
    
    args = parser.parse_args()
    
    # 初始化评估器
    evaluator = AudioQualityEvaluator(args.metrics)
    
    # 评估音频质量
    scores = evaluator.evaluate(
        test_path=args.test,
        reference_path=args.reference,
        window=args.window,
        score_rate=args.sample_rate,
        return_mean=args.mean
    )
    
    # 打印结果
    print("评估结果:")
    pprint.pprint(scores)
    
    # 保存结果
    evaluator.save_results(scores, args.output)


if __name__ == '__main__':
    main()
```

## 使用示例

### 命令行使用

1. 评估单个音频文件（使用所有指标）：

```bash
python audio_quality_evaluator.py --test audios/noisy.wav --reference audios/clean.wav --output results/single_file_results.json
```

2. 评估目录中的多个音频文件并计算平均分数：

```bash
python audio_quality_evaluator.py --test audios/noisy/ --reference audios/clean/ --output results/directory_results.json --mean
```

3. 只使用特定指标（如 DNSMOS 和 SRMR，这些不需要参考音频）：

```bash
python audio_quality_evaluator.py --test audios/noisy.wav --output results/no_reference_results.json --metrics DNSMOS SRMR
```

4. 指定采样率和窗口大小：

```bash
python audio_quality_evaluator.py --test audios/noisy.wav --reference audios/clean.wav --output results/custom_params_results.json --sample_rate 44100 --window 2.0
```

### 在 Python 代码中使用

```python
from audio_quality_evaluator import AudioQualityEvaluator

# 初始化评估器（使用所有指标）
evaluator = AudioQualityEvaluator()

# 评估单个文件
single_file_scores = evaluator.evaluate(
    test_path='audios/noisy.wav',
    reference_path='audios/clean.wav'
)

# 保存结果
evaluator.save_results(single_file_scores, 'results/single_file_results.json')

# 评估目录并计算平均分数
directory_scores = evaluator.evaluate(
    test_path='audios/noisy/',
    reference_path='audios/clean/',
    return_mean=True
)

# 保存结果
evaluator.save_results(directory_scores, 'results/directory_results.json')

# 只使用不需要参考的指标
no_reference_evaluator = AudioQualityEvaluator(['DNSMOS', 'SRMR'])
no_reference_scores = no_reference_evaluator.evaluate(
    test_path='audios/noisy.wav'
)

# 保存结果
no_reference_evaluator.save_results(no_reference_scores, 'results/no_reference_results.json')
```

## 功能特点

1. **灵活的指标选择**：可以选择使用所有支持的指标或指定特定指标
2. **支持单文件和目录**：可以评估单个音频文件或目录中的多个文件
3. **智能处理参考音频**：自动识别哪些指标需要参考音频，哪些不需要
4. **结果保存为 JSON**：评估结果以结构化 JSON 格式保存，便于后续分析
5. **命令行和 API 接口**：同时支持命令行使用和在 Python 代码中调用

## 注意事项

1. 某些指标（如 PESQ、BSSEval、STOI 等）需要参考音频，而其他指标（如 DNSMOS、SRMR）不需要
2. 如果只使用不需要参考的指标，可以不提供 `reference_path`
3. 评估目录时，测试音频和参考音频的文件名应当匹配
4. 默认采样率为 16000Hz，可以根据需要调整
