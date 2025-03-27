

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

```python
#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import json
import argparse
import hashlib
import pickle
import multiprocessing
from pathlib import Path
import pprint
from typing import List, Dict, Union, Optional
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
from speechscore import SpeechScore

class AudioQualityEvaluator:
    """音频质量评估工具，基于 SpeechScore 封装，支持并行处理和缓存"""
    
    def __init__(self, 
                metrics: Optional[List[str]] = None, 
                preload_models: bool = True,
                use_cache: bool = True, 
                cache_dir: str = '.cache'):
        """
        初始化评估器
        
        Args:
            metrics: 要评估的指标列表，如果为 None，则使用所有支持的指标
            preload_models: 是否预加载模型（如DNSMOS）
            use_cache: 是否使用缓存
            cache_dir: 缓存目录
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
        
        # 缓存设置
        self.use_cache = use_cache
        self.cache_dir = cache_dir
        
        if use_cache:
            os.makedirs(cache_dir, exist_ok=True)
        
        # 预加载模型以避免每次评估时的加载延迟
        if preload_models and 'DNSMOS' in metrics:
            try:
                import numpy as np
                dummy_audio = np.zeros(16000, dtype=np.float32)  # 1秒的静音
                # 注意：这里假设SpeechScore有_preload_dnsmos_model方法
                # 如果没有，可能需要修改或删除这部分代码
                try:
                    self.speech_score._preload_dnsmos_model(dummy_audio)
                except AttributeError:
                    # 如果没有这个方法，尝试通过评估一个小样本来预加载
                    self.speech_score(test_path=dummy_audio, reference_path=None)
            except:
                print("预加载DNSMOS模型失败，将在首次使用时加载")
    
    def _get_cache_key(self, test_path: str, reference_path: Optional[str], window: Optional[float], score_rate: int):
        """生成缓存键"""
        # 获取文件修改时间作为缓存的一部分
        test_mtime = os.path.getmtime(test_path) if os.path.exists(test_path) else 0
        ref_mtime = os.path.getmtime(reference_path) if reference_path and os.path.exists(reference_path) else 0
        
        # 组合所有参数生成唯一键
        key_parts = [
            test_path, str(test_mtime),
            str(reference_path), str(ref_mtime),
            str(window), str(score_rate),
            ','.join(sorted(self.metrics))
        ]
        
        key_str = '_'.join(key_parts)
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def _get_cache_path(self, cache_key: str):
        """获取缓存文件路径"""
        return os.path.join(self.cache_dir, f"{cache_key}.pkl")
    
    def _load_from_cache(self, cache_key: str):
        """从缓存加载结果"""
        cache_path = self._get_cache_path(cache_key)
        if os.path.exists(cache_path):
            try:
                with open(cache_path, 'rb') as f:
                    return pickle.load(f)
            except:
                print(f"读取缓存 {cache_path} 失败")
        return None
    
    def _save_to_cache(self, cache_key: str, data):
        """保存结果到缓存"""
        cache_path = self._get_cache_path(cache_key)
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(data, f)
        except:
            print(f"保存缓存 {cache_path} 失败")
    
    def _preprocess_audio(self, audio_path: str, target_sr: int = 16000):
        """
        预处理音频文件，优化加载速度
        
        Args:
            audio_path: 音频文件路径
            target_sr: 目标采样率
            
        Returns:
            预处理后的音频数据
        """
        try:
            import soundfile as sf
            import librosa
            
            try:
                # 使用 soundfile 加载音频（通常比 librosa 快）
                audio, sr = sf.read(audio_path)
                
                # 如果需要重采样
                if sr != target_sr:
                    # 使用 librosa 的快速重采样
                    audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
                    
                return audio, target_sr
            except:
                # 如果 soundfile 失败，回退到 librosa
                print(f"使用 soundfile 加载 {audio_path} 失败，尝试使用 librosa")
                audio, sr = librosa.load(audio_path, sr=target_sr)
                return audio, sr
        except ImportError:
            print("未安装 soundfile 或 librosa，无法预处理音频")
            return None, None
        
    def evaluate_file(self, 
                     test_file: str, 
                     reference_file: Optional[str] = None, 
                     window: Optional[float] = None, 
                     score_rate: int = 16000) -> Dict:
        """评估单个音频文件，支持缓存"""
        # 检查缓存
        if self.use_cache:
            cache_key = self._get_cache_key(test_file, reference_file, window, score_rate)
            cached_result = self._load_from_cache(cache_key)
            if cached_result is not None:
                return cached_result
        
        # 检查是否只使用不需要参考的指标
        non_reference_metrics = ['DNSMOS', 'SRMR']
        only_non_reference = all(metric in non_reference_metrics for metric in self.metrics)
        
        # 如果只使用不需要参考的指标，reference_file 可以为 None
        if reference_file is None and not only_non_reference:
            filtered_metrics = [m for m in self.metrics if m in non_reference_metrics]
            if not filtered_metrics:
                return {"error": "没有可用的指标！请提供参考音频或选择不需要参考的指标。"}
            temp_evaluator = SpeechScore(filtered_metrics)
            scores = temp_evaluator(test_path=test_file, reference_path=None, 
                                   window=window, score_rate=score_rate, return_mean=False)
        else:
            # 正常评估
            scores = self.speech_score(test_path=test_file, reference_path=reference_file, 
                                      window=window, score_rate=score_rate, return_mean=False)
        
        # 保存到缓存
        if self.use_cache:
            self._save_to_cache(cache_key, scores)
            
        return scores
        
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
    
    def evaluate_parallel(self, 
                         test_path: str, 
                         reference_path: Optional[str] = None, 
                         window: Optional[float] = None, 
                         score_rate: int = 16000, 
                         return_mean: bool = False,
                         max_workers: Optional[int] = None) -> Dict:
        """并行评估多个音频文件"""
        if os.path.isfile(test_path):
            # 单个文件直接评估
            return self.evaluate_file(test_path, reference_path, window, score_rate)
        
        # 处理目录
        test_files = [f for f in os.listdir(test_path) if f.endswith(('.wav', '.flac'))]
        results = {}
        
        # 如果没有指定最大工作进程数，使用CPU核心数
        if max_workers is None:
            max_workers = multiprocessing.cpu_count()
        
        # 准备参考文件路径
        ref_files = {}
        if reference_path and os.path.isdir(reference_path):
            for test_file in test_files:
                ref_file = os.path.join(reference_path, test_file)
                if os.path.exists(ref_file):
                    ref_files[test_file] = ref_file
                else:
                    print(f"警告: 找不到对应的参考文件 {ref_file}")
        
        # 创建部分函数，固定一些参数
        evaluate_func = partial(
            self._process_single_file,
            test_dir=test_path,
            reference_files=ref_files,
            window=window,
            score_rate=score_rate
        )
        
        # 并行处理
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(evaluate_func, test_file): test_file for test_file in test_files}
            
            for future in as_completed(futures):
                test_file = futures[future]
                try:
                    file_result = future.result()
                    results[test_file] = file_result
                except Exception as e:
                    print(f"处理文件 {test_file} 时出错: {str(e)}")
                    results[test_file] = {"error": str(e)}
        
        # 计算平均分数（如果需要）
        if return_mean and results:
            mean_scores = self._calculate_mean_scores(results)
            results["Mean_Score"] = mean_scores
            
        return results
    
    def _process_single_file(self, test_file, test_dir, reference_files, window, score_rate):
        """处理单个文件的辅助函数（用于并行处理）"""
        test_file_path = os.path.join(test_dir, test_file)
        ref_file_path = reference_files.get(test_file) if reference_files else None
        
        return self.evaluate_file(test_file_path, ref_file_path, window, score_rate)
    
    def _calculate_mean_scores(self, results):
        """计算所有文件的平均分数"""
        mean_scores = {}
        
        # 跳过可能的错误结果和Mean_Score本身
        valid_results = {k: v for k, v in results.items() 
                         if isinstance(v, dict) and "error" not in v and k != "Mean_Score"}
        
        if not valid_results:
            return {}
            
        # 遍历第一个结果来获取指标结构
        first_result = next(iter(valid_results.values()))
        
        # 初始化平均值字典
        for metric, value in first_result.items():
            if isinstance(value, dict):
                mean_scores[metric] = {}
                for sub_metric, sub_value in value.items():
                    if isinstance(sub_value, dict):
                        mean_scores[metric][sub_metric] = {}
                        for sub_sub_metric, sub_sub_value in sub_value.items():
                            mean_scores[metric][sub_metric][sub_sub_metric] = 0
                    else:
                        mean_scores[metric][sub_metric] = 0
            else:
                mean_scores[metric] = 0
        
        # 累加所有值
        for result in valid_results.values():
            for metric, value in result.items():
                if metric in mean_scores:
                    if isinstance(value, dict):
                        for sub_metric, sub_value in value.items():
                            if sub_metric in mean_scores[metric]:
                                if isinstance(sub_value, dict):
                                    # 处理嵌套字典
                                    for sub_sub_metric, sub_sub_value in sub_value.items():
                                        if sub_sub_metric in mean_scores[metric][sub_metric]:
                                            mean_scores[metric][sub_metric][sub_sub_metric] += sub_sub_value
                                else:
                                    mean_scores[metric][sub_metric] += sub_value
                    else:
                        mean_scores[metric] += value
        
        # 计算平均值
        count = len(valid_results)
        for metric, value in mean_scores.items():
            if isinstance(value, dict):
                for sub_metric, sub_value in value.items():
                    if isinstance(sub_value, dict):
                        for sub_sub_metric in sub_value:
                            mean_scores[metric][sub_metric][sub_sub_metric] /= count
                    else:
                        mean_scores[metric][sub_metric] /= count
            else:
                mean_scores[metric] /= count
                
        return mean_scores
    
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
    parser.add_argument('--parallel', action='store_true', help='是否使用并行处理')
    parser.add_argument('--workers', type=int, help='并行处理的最大工作进程数')
    parser.add_argument('--no-cache', action='store_true', help='禁用缓存')
    parser.add_argument('--cache-dir', default='.cache', help='缓存目录')
    parser.add_argument('--fast', action='store_true', help='使用快速模式（只计算轻量级指标）')
    
    args = parser.parse_args()
    
    # 快速模式下使用轻量级指标
    if args.fast:
        # 选择计算速度较快的指标
        metrics = ['SRMR', 'SNR', 'SSNR']
    else:
        metrics = args.metrics
    
    # 初始化评估器
    evaluator = AudioQualityEvaluator(
        metrics=metrics,
        preload_models=True,
        use_cache=not args.no_cache,
        cache_dir=args.cache_dir
    )
    
    # 评估音频质量
    if args.parallel and os.path.isdir(args.test):
        print(f"使用并行处理评估目录: {args.test}")
        scores = evaluator.evaluate_parallel(
            test_path=args.test,
            reference_path=args.reference,
            window=args.window,
            score_rate=args.sample_rate,
            return_mean=args.mean,
            max_workers=args.workers
        )
    else:
        if os.path.isdir(args.test):
            print(f"评估目录: {args.test}")
        else:
            print(f"评估文件: {args.test}")
        
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
    print(f"完成评估，结果已保存到: {args.output}")


if __name__ == '__main__':
    main()
```
