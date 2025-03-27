
# CosyVoice2 多样化音频生成代码

为了实现 DPO 训练中的数据采样逻辑，我将帮您封装一个函数，通过调整采样参数来生成多样化的音频样本。这个函数将基于给定的文本和参考音频，使用不同的采样参数生成多个音频样本。

```python
import torch
import torchaudio
import numpy as np
from typing import List, Dict, Tuple
import os
from cosyvoice.cli.cosyvoice import CosyVoice2
from cosyvoice.utils.file_utils import logging
import yaml
import copy

def generate_diverse_samples(
    model_path: str,
    text: str,
    reference_audio_path: str,
    output_dir: str,
    num_samples: int = 5,
    sampling_params_list: List[Dict] = None,
    stream: bool = False,
    fp16: bool = False
) -> List[str]:
    """
    生成多样化的音频样本，通过调整采样参数
    
    Args:
        model_path: CosyVoice2 模型路径
        text: 要合成的文本
        reference_audio_path: 参考音频路径（说话人音频）
        output_dir: 输出目录
        num_samples: 要生成的样本数量
        sampling_params_list: 采样参数列表，每个元素是一个字典，包含采样参数
        stream: 是否使用流式处理
        fp16: 是否使用半精度计算
        
    Returns:
        生成的音频文件路径列表
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 加载参考音频
    reference_audio, sr = torchaudio.load(reference_audio_path)
    if sr != 16000:
        # 重采样到16kHz
        resampler = torchaudio.transforms.Resample(sr, 16000)
        reference_audio = resampler(reference_audio)
    reference_audio_16k = reference_audio.squeeze(0)
    
    # 如果没有提供采样参数列表，则生成默认的多样化参数
    if sampling_params_list is None:
        sampling_params_list = generate_diverse_sampling_params(num_samples)
    
    # 确保有足够的参数集
    if len(sampling_params_list) < num_samples:
        logging.warning(f"提供的采样参数数量({len(sampling_params_list)})少于请求的样本数量({num_samples})，将重复使用参数")
        # 复制参数直到达到所需数量
        while len(sampling_params_list) < num_samples:
            sampling_params_list.extend(sampling_params_list[:num_samples-len(sampling_params_list)])
    
    output_paths = []
    
    # 对每组采样参数生成一个样本
    for i, sampling_params in enumerate(sampling_params_list[:num_samples]):
        # 修改模型配置文件中的采样参数
        model_config_path = os.path.join(model_path, "cosyvoice2.yaml")
        modified_config_path = modify_sampling_params(model_config_path, sampling_params, i)
        
        # 使用修改后的配置初始化模型
        cosyvoice = CosyVoice2(model_path, load_jit=False, load_trt=False, fp16=fp16, 
                              config_path=modified_config_path)
        
        # 生成音频
        output_path = os.path.join(output_dir, f"sample_{i}.wav")
        
        # 使用零样本合成
        for j, output in enumerate(cosyvoice.inference_zero_shot(
            text, 
            "", # 空提示文本
            reference_audio_16k, 
            stream=stream,
            text_frontend=False  # 使用与示例代码相同的设置
        )):
            if j > 0:
                logging.warning(f"生成了多个音频片段，只保留第一个")
                break
            
            torchaudio.save(output_path, output['tts_speech'], cosyvoice.sample_rate)
            output_paths.append(output_path)
        
        # 释放模型内存
        del cosyvoice
        torch.cuda.empty_cache()
        
        # 删除临时配置文件
        if os.path.exists(modified_config_path):
            os.remove(modified_config_path)
    
    return output_paths

def generate_diverse_sampling_params(num_samples: int) -> List[Dict]:
    """
    生成多样化的采样参数
    
    Args:
        num_samples: 要生成的参数集数量
        
    Returns:
        采样参数列表
    """
    params_list = []
    
    # 基础参数
    base_params = {
        "top_p": 0.8,
        "top_k": 25,
        "win_size": 10,
        "tau_r": 0.1
    }
    
    # 添加基础参数
    params_list.append(base_params)
    
    # 生成更多样化的参数
    if num_samples > 1:
        # 高多样性参数
        params_list.append({
            "top_p": 0.95,
            "top_k": 50,
            "win_size": 15,
            "tau_r": 0.2
        })
    
    if num_samples > 2:
        # 低多样性参数
        params_list.append({
            "top_p": 0.6,
            "top_k": 10,
            "win_size": 5,
            "tau_r": 0.05
        })
    
    if num_samples > 3:
        # 中等多样性，偏向高温度
        params_list.append({
            "top_p": 0.9,
            "top_k": 30,
            "win_size": 12,
            "tau_r": 0.15
        })
    
    if num_samples > 4:
        # 中等多样性，偏向低温度
        params_list.append({
            "top_p": 0.7,
            "top_k": 20,
            "win_size": 8,
            "tau_r": 0.08
        })
    
    # 如果需要更多参数，随机生成
    while len(params_list) < num_samples:
        params_list.append({
            "top_p": np.clip(np.random.normal(0.8, 0.1), 0.5, 0.99),
            "top_k": int(np.clip(np.random.normal(25, 10), 5, 100)),
            "win_size": int(np.clip(np.random.normal(10, 3), 3, 20)),
            "tau_r": np.clip(np.random.normal(0.1, 0.05), 0.01, 0.3)
        })
    
    return params_list

def modify_sampling_params(config_path: str, sampling_params: Dict, index: int) -> str:
    """
    修改模型配置文件中的采样参数
    
    Args:
        config_path: 原始配置文件路径
        sampling_params: 新的采样参数
        index: 样本索引，用于生成唯一的配置文件名
        
    Returns:
        修改后的配置文件路径
    """
    # 读取原始配置
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 修改采样参数
    if 'llm' in config and 'sampling' in config['llm']:
        # 深拷贝配置以避免修改原始对象
        new_config = copy.deepcopy(config)
        
        # 更新采样参数
        for key, value in sampling_params.items():
            if key in new_config['llm']['sampling']:
                new_config['llm']['sampling'][key] = value
    else:
        logging.warning("配置文件中未找到采样参数路径，将使用原始配置")
        new_config = config
    
    # 保存修改后的配置
    modified_config_path = f"{os.path.splitext(config_path)[0]}_modified_{index}.yaml"
    with open(modified_config_path, 'w', encoding='utf-8') as f:
        yaml.dump(new_config, f)
    
    return modified_config_path

def dpo_sample_generation(
    model_path: str,
    text_list: List[str],
    reference_audio_path: str,
    output_base_dir: str,
    samples_per_text: int = 5,
    sampling_params_list: List[Dict] = None,
    stream: bool = False,
    fp16: bool = False
) -> Dict[str, List[str]]:
    """
    为DPO训练生成多样化的音频样本
    
    Args:
        model_path: CosyVoice2 模型路径
        text_list: 要合成的文本列表
        reference_audio_path: 参考音频路径（说话人音频）
        output_base_dir: 输出基础目录
        samples_per_text: 每个文本要生成的样本数量
        sampling_params_list: 采样参数列表
        stream: 是否使用流式处理
        fp16: 是否使用半精度计算
        
    Returns:
        文本到音频路径列表的映射
    """
    results = {}
    
    for i, text in enumerate(text_list):
        # 为每个文本创建一个子目录
        text_dir = os.path.join(output_base_dir, f"text_{i}")
        
        # 生成样本
        audio_paths = generate_diverse_samples(
            model_path=model_path,
            text=text,
            reference_audio_path=reference_audio_path,
            output_dir=text_dir,
            num_samples=samples_per_text,
            sampling_params_list=sampling_params_list,
            stream=stream,
            fp16=fp16
        )
        
        results[text] = audio_paths
        logging.info(f"为文本 {i} 生成了 {len(audio_paths)} 个样本")
    
    return results

# 使用示例
if __name__ == "__main__":
    # 示例文本列表
    texts = [
        "这是一个用于测试的句子，希望能生成多样化的语音样本。",
        "人工智能技术正在快速发展，语音合成质量越来越高。"
    ]
    
    # 生成样本
    results = dpo_sample_generation(
        model_path="pretrained_models/CosyVoice2-0.5B",
        text_list=texts,
        reference_audio_path="./asset/zero_shot_prompt.wav",
        output_base_dir="./dpo_samples",
        samples_per_text=5
    )
    
    # 打印结果
    for text, paths in results.items():
        print(f"文本: {text}")
        print(f"生成的样本: {paths}")
        print("---")
```

## 代码说明

1. **主要函数**:
   - `generate_diverse_samples`: 为单个文本生成多样化的音频样本
   - `generate_diverse_sampling_params`: 生成多样化的采样参数
   - `modify_sampling_params`: 修改模型配置文件中的采样参数
   - `dpo_sample_generation`: 为多个文本生成样本，用于DPO训练

2. **采样参数变化**:
   - 调整 `top_p`: 控制采样概率分布的截断阈值（0.6-0.95）
   - 调整 `top_k`: 控制每步保留的最高概率标记数（10-50）
   - 调整 `win_size`: 控制滑动窗口大小（5-15）
   - 调整 `tau_r`: 控制重复惩罚的强度（0.05-0.2）

3. **实现细节**:
   - 通过修改配置文件来改变采样参数，而不是直接修改模型内部参数
   - 每次生成完成后释放模型内存，避免GPU内存溢出
   - 支持批量处理多个文本
   - 为每个文本创建单独的输出目录

4. **使用方法**:
   - 提供模型路径、文本列表、参考音频和输出目录
   - 可以自定义每个文本生成的样本数量
   - 可以提供自定义的采样参数列表，或使用自动生成的多样化参数

## 使用建议

1. 对于DPO训练，建议生成5-10个样本，以获得足够的多样性
2. 可以根据需要调整采样参数的范围，以控制生成样本的多样性程度
3. 如果GPU内存有限，可以减少每批处理的文本数量
4. 生成样本后，可以使用人工或自动评分系统为每个样本打分，然后选择最好和最差的样本用于DPO训练

这个代码框架可以帮助您为CosyVoice2模型的DPO训练生成多样化的音频样本，通过调整采样参数来探索模型的不同生成可能性。






# CosyVoice2 多样化音频生成代码 - 优化版

以下是根据您的需求优化后的完整代码，支持从 Excel 文件读取数据，并按照指定的目录结构组织生成的音频文件：

```python
import torch
import torchaudio
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
import os
from cosyvoice.cli.cosyvoice import CosyVoice2
from cosyvoice.utils.file_utils import logging
import yaml
import copy
import argparse
from tqdm import tqdm

def generate_diverse_samples(
    model_path: str,
    text: str,
    reference_audio_path: str,
    output_dir: str,
    text_id: str,
    num_samples: int = 5,
    sampling_params_list: Optional[List[Dict]] = None,
    stream: bool = False,
    fp16: bool = False
) -> List[str]:
    """
    生成多样化的音频样本，通过调整采样参数
    
    Args:
        model_path: CosyVoice2 模型路径
        text: 要合成的文本
        reference_audio_path: 参考音频路径（说话人音频）
        output_dir: 输出目录
        text_id: 文本唯一标识符
        num_samples: 要生成的样本数量
        sampling_params_list: 采样参数列表，每个元素是一个字典，包含采样参数
        stream: 是否使用流式处理
        fp16: 是否使用半精度计算
        
    Returns:
        生成的音频文件路径列表
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 加载参考音频
    reference_audio, sr = torchaudio.load(reference_audio_path)
    if sr != 16000:
        # 重采样到16kHz
        resampler = torchaudio.transforms.Resample(sr, 16000)
        reference_audio = resampler(reference_audio)
    reference_audio_16k = reference_audio.squeeze(0)
    
    # 如果没有提供采样参数列表，则生成默认的多样化参数
    if sampling_params_list is None:
        sampling_params_list = generate_diverse_sampling_params(num_samples)
    
    # 确保有足够的参数集
    if len(sampling_params_list) < num_samples:
        logging.warning(f"提供的采样参数数量({len(sampling_params_list)})少于请求的样本数量({num_samples})，将重复使用参数")
        # 复制参数直到达到所需数量
        while len(sampling_params_list) < num_samples:
            sampling_params_list.extend(sampling_params_list[:num_samples-len(sampling_params_list)])
    
    output_paths = []
    
    # 初始化模型（只初始化一次）
    cosyvoice = CosyVoice2(model_path, load_jit=False, load_trt=False, fp16=fp16)
    
    # 对每组采样参数生成一个样本
    for i, sampling_params in enumerate(sampling_params_list[:num_samples]):
        # 设置输出路径
        output_path = os.path.join(output_dir, f"{text_id}_{i}.wav")
        
        # 使用零样本合成
        try:
            # 注意：这里我们没有修改配置文件，而是直接使用模型
            # 在实际应用中，您可能需要找到一种方法来动态修改模型的采样参数
            for j, output in enumerate(cosyvoice.inference_zero_shot(
                text, 
                "", # 空提示文本
                reference_audio_16k, 
                stream=stream,
                text_frontend=False,  # 使用与示例代码相同的设置
                sampling_params=sampling_params  # 假设模型支持直接传入采样参数
            )):
                if j > 0:
                    logging.warning(f"生成了多个音频片段，只保留第一个")
                    break
                
                torchaudio.save(output_path, output['tts_speech'], cosyvoice.sample_rate)
                output_paths.append(output_path)
        except Exception as e:
            logging.error(f"生成样本 {i} 时出错: {str(e)}")
    
    # 释放模型内存
    del cosyvoice
    torch.cuda.empty_cache()
    
    return output_paths

def generate_diverse_sampling_params(num_samples: int) -> List[Dict]:
    """
    生成多样化的采样参数
    
    Args:
        num_samples: 要生成的参数集数量
        
    Returns:
        采样参数列表
    """
    params_list = []
    
    # 基础参数
    base_params = {
        "top_p": 0.8,
        "top_k": 25,
        "win_size": 10,
        "tau_r": 0.1
    }
    
    # 添加基础参数
    params_list.append(base_params)
    
    # 生成更多样化的参数
    if num_samples > 1:
        # 高多样性参数
        params_list.append({
            "top_p": 0.95,
            "top_k": 50,
            "win_size": 15,
            "tau_r": 0.2
        })
    
    if num_samples > 2:
        # 低多样性参数
        params_list.append({
            "top_p": 0.6,
            "top_k": 10,
            "win_size": 5,
            "tau_r": 0.05
        })
    
    if num_samples > 3:
        # 中等多样性，偏向高温度
        params_list.append({
            "top_p": 0.9,
            "top_k": 30,
            "win_size": 12,
            "tau_r": 0.15
        })
    
    if num_samples > 4:
        # 中等多样性，偏向低温度
        params_list.append({
            "top_p": 0.7,
            "top_k": 20,
            "win_size": 8,
            "tau_r": 0.08
        })
    
    # 如果需要更多参数，随机生成
    while len(params_list) < num_samples:
        params_list.append({
            "top_p": np.clip(np.random.normal(0.8, 0.1), 0.5, 0.99),
            "top_k": int(np.clip(np.random.normal(25, 10), 5, 100)),
            "win_size": int(np.clip(np.random.normal(10, 3), 3, 20)),
            "tau_r": np.clip(np.random.normal(0.1, 0.05), 0.01, 0.3)
        })
    
    return params_list

def process_excel_for_dpo(
    model_path: str,
    excel_path: str,
    output_base_dir: str,
    samples_per_text: int = 5,
    sampling_params_list: Optional[List[Dict]] = None,
    stream: bool = False,
    fp16: bool = False
) -> Dict[str, Dict[str, List[str]]]:
    """
    处理Excel文件，为DPO训练生成多样化的音频样本
    
    Args:
        model_path: CosyVoice2 模型路径
        excel_path: Excel文件路径，包含text_id, text, speaker, reference_audio_path列
        output_base_dir: 输出基础目录
        samples_per_text: 每个文本要生成的样本数量
        sampling_params_list: 采样参数列表
        stream: 是否使用流式处理
        fp16: 是否使用半精度计算
        
    Returns:
        嵌套字典，格式为 {speaker: {text_id: [audio_paths]}}
    """
    # 创建输出基础目录
    os.makedirs(output_base_dir, exist_ok=True)
    
    # 读取Excel文件
    if excel_path.endswith('.xlsx') or excel_path.endswith('.xls'):
        df = pd.read_excel(excel_path)
    elif excel_path.endswith('.csv'):
        df = pd.read_csv(excel_path)
    else:
        raise ValueError(f"不支持的文件格式: {excel_path}")
    
    # 验证必要的列是否存在
    required_columns = ['text_id', 'text', 'speaker', 'reference_audio_path']
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Excel文件中缺少必要的列: {col}")
    
    results = {}
    
    # 按speaker分组处理
    for speaker, group in df.groupby('speaker'):
        speaker_dir = os.path.join(output_base_dir, speaker)
        os.makedirs(speaker_dir, exist_ok=True)
        
        speaker_results = {}
        
        # 处理每个文本
        for _, row in tqdm(group.iterrows(), total=len(group), desc=f"处理说话人 {speaker}"):
            text_id = str(row['text_id'])
            text = row['text']
            reference_audio_path = row['reference_audio_path']
            
            # 创建text_id目录
            text_dir = os.path.join(speaker_dir, text_id)
            
            # 生成样本
            try:
                audio_paths = generate_diverse_samples(
                    model_path=model_path,
                    text=text,
                    reference_audio_path=reference_audio_path,
                    output_dir=text_dir,
                    text_id=text_id,
                    num_samples=samples_per_text,
                    sampling_params_list=sampling_params_list,
                    stream=stream,
                    fp16=fp16
                )
                
                speaker_results[text_id] = audio_paths
                logging.info(f"为说话人 {speaker} 的文本 {text_id} 生成了 {len(audio_paths)} 个样本")
            except Exception as e:
                logging.error(f"处理说话人 {speaker} 的文本 {text_id} 时出错: {str(e)}")
        
        results[speaker] = speaker_results
    
    return results

def patch_cosyvoice2_for_sampling_params():
    """
    为CosyVoice2类添加支持直接传入采样参数的功能
    
    注意：这是一个猴子补丁，用于修改CosyVoice2类的行为
    在实际应用中，您可能需要直接修改CosyVoice2的源代码
    """
    # 保存原始的inference_zero_shot方法
    original_inference_zero_shot = CosyVoice2.inference_zero_shot
    
    # 定义新的方法，支持sampling_params参数
    def patched_inference_zero_shot(self, tts_text, prompt_text, prompt_speech_16k, stream=False, speed=1.0, text_frontend=True, sampling_params=None):
        # 如果提供了sampling_params，临时修改模型的采样参数
        if sampling_params is not None:
            # 保存原始参数
            original_params = {}
            if hasattr(self.model.llm, 'sampling'):
                for key, value in sampling_params.items():
                    if hasattr(self.model.llm.sampling, key):
                        original_params[key] = getattr(self.model.llm.sampling, key)
                        setattr(self.model.llm.sampling, key, value)
        
        # 调用原始方法
        result = original_inference_zero_shot(self, tts_text, prompt_text, prompt_speech_16k, stream, speed, text_frontend)
        
        # 如果修改了参数，恢复原始参数
        if sampling_params is not None and hasattr(self.model.llm, 'sampling'):
            for key, value in original_params.items():
                if hasattr(self.model.llm.sampling, key):
                    setattr(self.model.llm.sampling, key, value)
        
        return result
    
    # 替换方法
    CosyVoice2.inference_zero_shot = patched_inference_zero_shot

def main():
    parser = argparse.ArgumentParser(description='为DPO训练生成多样化的CosyVoice2音频样本')
    parser.add_argument('--model_path', type=str, required=True, help='CosyVoice2模型路径')
    parser.add_argument('--excel_path', type=str, required=True, help='Excel文件路径')
    parser.add_argument('--output_dir', type=str, required=True, help='输出目录')
    parser.add_argument('--samples_per_text', type=int, default=5, help='每个文本生成的样本数量')
    parser.add_argument('--stream', action='store_true', help='是否使用流式处理')
    parser.add_argument('--fp16', action='store_true', help='是否使用半精度计算')
    
    args = parser.parse_args()
    
    # 应用猴子补丁，使CosyVoice2支持直接传入采样参数
    patch_cosyvoice2_for_sampling_params()
    
    # 处理Excel文件
    results = process_excel_for_dpo(
        model_path=args.model_path,
        excel_path=args.excel_path,
        output_base_dir=args.output_dir,
        samples_per_text=args.samples_per_text,
        stream=args.stream,
        fp16=args.fp16
    )
    
    # 打印结果摘要
    print("\n生成结果摘要:")
    for speaker, speaker_results in results.items():
        print(f"说话人 {speaker}:")
        print(f"  处理的文本数量: {len(speaker_results)}")
        total_samples = sum(len(paths) for paths in speaker_results.values())
        print(f"  生成的样本总数: {total_samples}")
    
    print(f"\n所有样本已保存到: {args.output_dir}")

if __name__ == "__main__":
    main()
```

## 使用方法

1. 将代码保存为 `generate_dpo_samples.py`
2. 准备一个包含必要列的Excel文件（text_id, text, speaker, reference_audio_path）
3. 运行以下命令：

```bash
python generate_dpo_samples.py --model_path pretrained_models/CosyVoice2-0.5B --excel_path your_data.xlsx --output_dir ./dpo_samples --samples_per_text 5
```

## 代码说明

1. **主要功能**:
   - 从Excel文件读取数据
   - 按照指定的目录结构组织输出（speaker/text_id/text_id_数字编号.wav）
   - 为每个文本生成多个具有不同采样参数的音频样本

2. **目录结构**:
   ```
   output_dir/
   ├── speaker1/
   │   ├── text_id1/
   │   │   ├── text_id1_0.wav
   │   │   ├── text_id1_1.wav
   │   │   └── ...
   │   ├── text_id2/
   │   │   ├── text_id2_0.wav
   │   │   └── ...
   │   └── ...
   ├── speaker2/
   │   └── ...
   └── ...
   ```

3. **关键改进**:
   - 添加了猴子补丁，使CosyVoice2支持直接传入采样参数
   - 按照speaker和text_id组织输出目录结构
   - 支持从Excel/CSV文件读取数据
   - 添加了命令行参数支持
   - 使用tqdm显示进度条
   - 增强了错误处理和日志记录

4. **注意事项**:
   - 猴子补丁是一种临时解决方案，在实际应用中，您可能需要直接修改CosyVoice2的源代码
   - 代码假设CosyVoice2类的inference_zero_shot方法可以接受sampling_params参数，如果不支持，您需要修改源代码

## 使用建议

1. 确保Excel文件中的reference_audio_path列包含有效的音频文件路径
2. 对于大量数据，考虑分批处理，以避免内存问题
3. 如果遇到GPU内存不足的问题，可以尝试使用fp16模式（添加--fp16参数）
4. 生成样本后，您可以手动或使用自动评分系统为每个样本打分，然后选择最好和最差的样本用于DPO训练

这个优化版本的代码应该能够满足您的需求，按照指定的目录结构组织生成的音频文件，并支持从Excel文件读取数据。

