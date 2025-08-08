







# CosyVoice2 DPO训练数据准备完整流程



## processor_dpo.py 分析

`processor_dpo.py`文件定义了一系列用于DPO训练数据处理的函数，主要包括：

1. **parquet_opener**：打开Parquet文件并读取数据
2. **filter**：根据长度等条件过滤样本
3. **resample**：重采样音频到指定采样率
4. **truncate**：截断或填充音频到固定长度
5. **compute_fbank**：计算滤波器组特征
6. **compute_f0**：计算基频（F0）
7. **parse_embedding**：处理嵌入向量
8. **tokenize**：文本分词
9. **shuffle**：随机打乱数据
10. **sort**：按长度排序数据
11. **batch**：将数据分批
12. **padding**：对批次数据进行填充

与普通处理器的主要区别在于：

- 支持DPO模式，处理`reject_speech_token`
- 适应DPO训练的特殊数据格式
- 在`padding`函数中添加了DPO相关的处理逻辑

## 完整的DPO训练数据准备流程

基于所有提供的代码，我将为您提供一个完整的DPO训练数据准备流程：

### 1. 准备目录结构

```bash
mkdir -p dpo_data/{raw,processed,samples}
```


### 2. 生成多样化样本

使用我们之前开发的代码生成多样化样本：

```python
#!/usr/bin/env python3
import argparse
import os
import pandas as pd
from tqdm import tqdm
import torch
import torchaudio
from generate_dpo_samples import process_excel_for_dpo

def main():
    parser = argparse.ArgumentParser(description="生成CosyVoice2 DPO训练样本")
    parser.add_argument("--model_path", type=str, required=True, help="CosyVoice2模型路径")
    parser.add_argument("--excel_path", type=str, required=True, help="Excel文件路径")
    parser.add_argument("--output_dir", type=str, required=True, help="输出目录")
    parser.add_argument("--samples_per_text", type=int, default=5, help="每个文本生成的样本数量")
    parser.add_argument("--fp16", action="store_true", help="是否使用半精度计算")
    
    args = parser.parse_args()
    
    # 生成多样化样本
    results = process_excel_for_dpo(
        model_path=args.model_path,
        excel_path=args.excel_path,
        output_base_dir=args.output_dir,
        samples_per_text=args.samples_per_text,
        fp16=args.fp16,
        text_frontend=False
    )
    
    print(f"样本生成完成，保存在 {args.output_dir}")

if __name__ == "__main__":
    main()
```


运行命令：

```bash
python generate_samples.py --model_path pretrained_models/CosyVoice2-0.5B --excel_path data.xlsx --output_dir dpo_data/samples --samples_per_text 5
```


### 3. 评估样本并选择最好和最差的样本

这一步可以通过人工评估或自动评分系统完成。以下是一个简单的脚本，用于记录人工评分：

```python
#!/usr/bin/env python3
import os
import json
import argparse
import pandas as pd
from tqdm import tqdm
import pygame
import time

def play_audio(file_path):
    """播放音频文件"""
    pygame.mixer.init()
    pygame.mixer.music.load(file_path)
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy():
        time.sleep(0.1)

def main():
    parser = argparse.ArgumentParser(description="评估CosyVoice2生成的样本")
    parser.add_argument("--samples_dir", type=str, required=True, help="样本目录")
    parser.add_argument("--output_file", type=str, required=True, help="评分结果输出文件")
    
    args = parser.parse_args()
    
    # 初始化评分结果
    if os.path.exists(args.output_file):
        with open(args.output_file, 'r') as f:
            scores = json.load(f)
    else:
        scores = {}
    
    # 遍历所有样本
    for speaker in os.listdir(args.samples_dir):
        speaker_dir = os.path.join(args.samples_dir, speaker)
        if not os.path.isdir(speaker_dir):
            continue
        
        if speaker not in scores:
            scores[speaker] = {}
        
        for text_id in os.listdir(speaker_dir):
            text_dir = os.path.join(speaker_dir, text_id)
            if not os.path.isdir(text_dir):
                continue
            
            if text_id not in scores[speaker]:
                scores[speaker][text_id] = {}
            
            # 获取所有音频文件
            audio_files = [f for f in os.listdir(text_dir) if f.endswith('.wav')]
            
            for audio_file in audio_files:
                sample_id = os.path.splitext(audio_file)[0]
                if sample_id in scores[speaker][text_id]:
                    continue
                
                audio_path = os.path.join(text_dir, audio_file)
                
                # 播放音频
                print(f"\n正在播放: {speaker}/{text_id}/{audio_file}")
                play_audio(audio_path)
                
                # 获取评分
                score = input("请为这个样本打分 (1-10): ")
                try:
                    score = float(score)
                    scores[speaker][text_id][sample_id] = score
                    
                    # 保存当前评分结果
                    with open(args.output_file, 'w') as f:
                        json.dump(scores, f, indent=2)
                except ValueError:
                    print("无效的评分，跳过")
    
    # 为每个text_id选择最好和最差的样本
    best_worst = {}
    for speaker in scores:
        best_worst[speaker] = {}
        for text_id in scores[speaker]:
            sample_scores = scores[speaker][text_id]
            if not sample_scores:
                continue
            
            best_sample = max(sample_scores.items(), key=lambda x: x[1])[0]
            worst_sample = min(sample_scores.items(), key=lambda x: x[1])[0]
            
            best_worst[speaker][text_id] = {
                "best": best_sample,
                "worst": worst_sample
            }
    
    # 保存最好和最差的样本结果
    with open(os.path.splitext(args.output_file)[0] + "_best_worst.json", 'w') as f:
        json.dump(best_worst, f, indent=2)
    
    print(f"评分完成，结果保存在 {args.output_file}")
    print(f"最好和最差的样本信息保存在 {os.path.splitext(args.output_file)[0]}_best_worst.json")

if __name__ == "__main__":
    main()
```


运行命令：

```bash
python evaluate_samples.py --samples_dir dpo_data/samples --output_file dpo_data/scores.json
```


### 4. 准备DPO训练数据

根据评估结果，准备DPO训练所需的数据文件：

```python
#!/usr/bin/env python3
import os
import json
import argparse
import shutil
import pandas as pd
from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser(description="准备CosyVoice2 DPO训练数据")
    parser.add_argument("--samples_dir", type=str, required=True, help="样本目录")
    parser.add_argument("--best_worst_file", type=str, required=True, help="最好和最差样本信息文件")
    parser.add_argument("--output_dir", type=str, required=True, help="输出目录")
    
    args = parser.parse_args()
    
    # 加载最好和最差样本信息
    with open(args.best_worst_file, 'r') as f:
        best_worst = json.load(f)
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 准备wav.scp, text, utt2spk文件
    with open(os.path.join(args.output_dir, "wav.scp"), 'w') as f_wav, \
         open(os.path.join(args.output_dir, "text"), 'w') as f_text, \
         open(os.path.join(args.output_dir, "utt2spk"), 'w') as f_spk:
        
        for speaker in best_worst:
            for text_id in best_worst[speaker]:
                # 获取最好和最差的样本
                best_sample = best_worst[speaker][text_id]["best"]
                worst_sample = best_worst[speaker][text_id]["worst"]
                
                # 构建文件路径
                best_path = os.path.join(args.samples_dir, speaker, text_id, f"{best_sample}.wav")
                worst_path = os.path.join(args.samples_dir, speaker, text_id, f"{worst_sample}.wav")
                
                if not os.path.exists(best_path) or not os.path.exists(worst_path):
                    print(f"警告: 样本文件不存在 - {best_path} 或 {worst_path}")
                    continue
                
                # 获取文本内容
                # 这里假设我们可以从Excel文件或其他地方获取文本内容
                # 在实际应用中，您需要根据实际情况获取文本
                text = text_id  # 这里简单地使用text_id作为文本，实际应用中需要替换
                
                # 写入wav.scp
                f_wav.write(f"{text_id}_best {best_path}\n")
                f_wav.write(f"{text_id}_worst {worst_path}\n")
                
                # 写入text
                f_text.write(f"{text_id}_best {text}\n")
                f_text.write(f"{text_id}_worst {text}\n")
                
                # 写入utt2spk
                f_spk.write(f"{text_id}_best {speaker}\n")
                f_spk.write(f"{text_id}_worst {speaker}\n")
    
    print(f"DPO训练数据准备完成，保存在 {args.output_dir}")

if __name__ == "__main__":
    main()
```


运行命令：

```bash
python prepare_dpo_data.py --samples_dir dpo_data/samples --best_worst_file dpo_data/scores_best_worst.json --output_dir dpo_data/raw
```


### 5. 提取嵌入向量和语音标记

```bash
# 提取嵌入向量
python tools/extract_embedding.py --dir dpo_data/raw --onnx_path pretrained_models/CosyVoice2-0.5B/campplus.onnx --num_thread 8

# 提取语音标记
python tools/extract_speech_token.py --dir dpo_data/raw --onnx_path pretrained_models/CosyVoice2-0.5B/speech_tokenizer_v2.onnx --num_thread 8
```


### 6. 创建拒绝语音标记文件

为DPO训练创建拒绝语音标记文件：

```python
#!/usr/bin/env python3
import os
import torch
import argparse

def main():
    parser = argparse.ArgumentParser(description="创建CosyVoice2 DPO拒绝语音标记文件")
    parser.add_argument("--raw_dir", type=str, required=True, help="原始数据目录")
    
    args = parser.parse_args()
    
    # 加载语音标记
    utt2speech_token = torch.load(os.path.join(args.raw_dir, "utt2speech_token.pt"))
    
    # 创建拒绝语音标记
    utt2reject_speech_token = {}
    
    for utt, token in utt2speech_token.items():
        if utt.endswith("_best"):
            # 找到对应的worst样本
            worst_utt = utt.replace("_best", "_worst")
            if worst_utt in utt2speech_token:
                # 将worst样本的语音标记作为best样本的拒绝语音标记
                utt2reject_speech_token[utt] = utt2speech_token[worst_utt]
        elif utt.endswith("_worst"):
            # 找到对应的best样本
            best_utt = utt.replace("_worst", "_best")
            if best_utt in utt2speech_token:
                # 将best样本的语音标记作为worst样本的拒绝语音标记
                utt2reject_speech_token[utt] = utt2speech_token[best_utt]
    
    # 保存拒绝语音标记
    torch.save(utt2reject_speech_token, os.path.join(args.raw_dir, "utt2reject_speech_token.pt"))
    
    print(f"拒绝语音标记文件创建完成，保存在 {os.path.join(args.raw_dir, 'utt2reject_speech_token.pt')}")
    print(f"共处理 {len(utt2reject_speech_token)} 个样本")

if __name__ == "__main__":
    main()
```


运行命令：

```bash
python create_reject_tokens.py --raw_dir dpo_data/raw
```


### 7. 打包数据

```bash
python tools/pack_data.py --src_dir dpo_data/raw --des_dir dpo_data/processed --num_utts_per_parquet 1000 --num_processes 4 --dpo
```


### 8. 完整的自动化脚本

以下是一个完整的自动化脚本，将上述所有步骤整合在一起：

```python
#!/usr/bin/env python3
import os
import argparse
import subprocess
import json
from datetime import datetime

def run_command(cmd, desc=None):
    """运行命令并显示描述"""
    if desc:
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {desc}")
    print(f"执行命令: {cmd}")
    subprocess.run(cmd, shell=True, check=True)

def prepare_dpo_data(args):
    """准备CosyVoice2 DPO训练数据"""
    # 创建目录
    os.makedirs(args.output_dir, exist_ok=True)
    samples_dir = os.path.join(args.output_dir, "samples")
    raw_dir = os.path.join(args.output_dir, "raw")
    processed_dir = os.path.join(args.output_dir, "processed")
    os.makedirs(samples_dir, exist_ok=True)
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(processed_dir, exist_ok=True)
    
    # 步骤1：生成多样化样本
    if not args.skip_generation:
        run_command(
            f"python generate_dpo_samples.py --model_path {args.model_path} "
            f"--excel_path {args.excel_path} --output_dir {samples_dir} "
            f"--samples_per_text {args.samples_per_text} {'--fp16' if args.fp16 else ''}",
            "生成多样化样本"
        )
    
    # 步骤2：评估样本
    if not args.skip_evaluation:
        if args.auto_evaluate:
            # 自动评估（这里需要实现自动评分逻辑）
            run_command(
                f"python auto_evaluate_samples.py --samples_dir {samples_dir} "
                f"--output_file {os.path.join(args.output_dir, 'scores.json')}",
                "自动评估样本"
            )
        else:
            # 人工评估
            run_command(
                f"python evaluate_samples.py --samples_dir {samples_dir} "
                f"--output_file {os.path.join(args.output_dir, 'scores.json')}",
                "人工评估样本"
            )
    
    # 步骤3：准备DPO训练数据
    if not args.skip_preparation:
        run_command(
            f"python prepare_dpo_data.py --samples_dir {samples_dir} "
            f"--best_worst_file {os.path.join(args.output_dir, 'scores_best_worst.json')} "
            f"--output_dir {raw_dir}",
            "准备DPO训练数据"
        )
    
    # 步骤4：提取嵌入向量
    if not args.skip_embedding:
        run_command(
            f"python tools/extract_embedding.py --dir {raw_dir} "
            f"--onnx_path {args.model_path}/campplus.onnx --num_thread {args.num_threads}",
            "提取嵌入向量"
        )
    
    # 步骤5：提取语音标记
    if not args.skip_speech_token:
        run_command(
            f"python tools/extract_speech_token.py --dir {raw_dir} "
            f"--onnx_path {args.model_path}/speech_tokenizer_v2.onnx --num_thread {args.num_threads}",
            "提取语音标记"
        )
    
    # 步骤6：创建拒绝语音标记
    if not args.skip_reject_token:
        run_command(
            f"python create_reject_tokens.py --raw_dir {raw_dir}",
            "创建拒绝语音标记"
        )
    
    # 步骤7：打包数据
    if not args.skip_packing:
        run_command(
            f"python tools/pack_data.py --src_dir {raw_dir} --des_dir {processed_dir} "
            f"--num_utts_per_parquet {args.num_utts_per_parquet} --num_processes {args.num_processes} --dpo",
            "打包数据"
        )
    
    # 步骤8：创建元数据
    metadata = {
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model_path": args.model_path,
        "excel_path": args.excel_path,
        "samples_per_text": args.samples_per_text,
        "num_utts_per_parquet": args.num_utts_per_parquet,
        "num_processes": args.num_processes,
        "num_threads": args.num_threads,
        "fp16": args.fp16
    }
    with open(os.path.join(args.output_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)
    
    print(f"DPO训练数据准备完成，保存在 {args.output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="准备CosyVoice2 DPO训练数据")
    parser.add_argument("--model_path", type=str, required=True, help="CosyVoice2模型路径")
    parser.add_argument("--excel_path", type=str, required=True, help="Excel文件路径")
    parser.add_argument("--output_dir", type=str, required=True, help="输出目录")
    parser.add_argument("--samples_per_text", type=int, default=5, help="每个文本生成的样本数量")
    parser.add_argument("--num_utts_per_parquet", type=int, default=1000, help="每个Parquet文件包含的语音片段数量")
    parser.add_argument("--num_processes", type=int, default=4, help="打包数据的进程数")
    parser.add_argument("--num_threads", type=int, default=8, help="提取特征的线程数")
    parser.add_argument("--fp16", action="store_true", help="是否使用半精度计算")
    parser.add_argument("--auto_evaluate", action="store_true", help="是否使用自动评估")
    parser.add_argument("--skip_generation", action="store_true", help="跳过样本生成步骤")
    parser.add_argument("--skip_evaluation", action="store_true", help="跳过样本评估步骤")
    parser.add_argument("--skip_preparation", action="store_true", help="跳过数据准备步骤")
    parser.add_argument("--skip_embedding", action="store_true", help="跳过嵌入向量提取步骤")
    parser.add_argument("--skip_speech_token", action="store_true", help="跳过语音标记提取步骤")
    parser.add_argument("--skip_reject_token", action="store_true", help="跳过拒绝语音标记创建步骤")
    parser.add_argument("--skip_packing", action="store_true", help="跳过数据打包步骤")
    
    args = parser.parse_args()
    
    prepare_dpo_data(args)
```


运行命令：

```bash
python prepare_all_dpo_data.py --model_path pretrained_models/CosyVoice2-0.5B --excel_path data.xlsx --output_dir dpo_data
```


## 总结与建议

1. **数据质量**：DPO训练的效果很大程度上取决于偏好数据的质量。确保"好"样本和"差"样本之间有明显但合理的差异。

2. **数据多样性**：确保训练数据覆盖各种语音风格、内容和说话人，以提高模型的泛化能力。

3. **自动评估**：对于大规模数据，可以考虑开发自动评分系统，例如使用预训练的语音质量评估模型。

4. **增量处理**：对于大量数据，可以考虑增量处理，避免一次性处理所有数据导致内存问题。

5. **数据平衡**：确保各个说话人和文本类型的数据分布均衡，避免模型偏向特定的说话人或内容。

6. **参数调优**：在DPO训练过程中，可能需要调整学习率、批次大小等参数，以获得最佳效果。

7. **验证集**：保留一部分数据作为验证集，用于监控训练过程和评估模型性能。

8. **模型保存**：定期保存训练中的模型检查点，以便在需要时恢复训练或选择最佳模型。

通过这个完整的流程，您应该能够准备高质量的DPO训练数据，为CosyVoice2模型的进一步优化提供基础。
