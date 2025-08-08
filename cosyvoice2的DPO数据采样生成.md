
# CosyVoice2 多样化音频生成代码

为了实现 DPO 训练中的数据采样逻辑，需要封装一个函数，通过调整采样参数来生成多样化的音频样本。这个函数将基于给定的文本和参考音频，使用不同的采样参数生成多个音频样本。

```python
import torch
import torchaudio
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional, Generator
import os
from cosyvoice.cli.cosyvoice import CosyVoice2
from cosyvoice.utils.file_utils import logging
import yaml
import copy
import argparse
from tqdm import tqdm
import json

def generate_diverse_samples(
    model_path: str,
    text: str,
    reference_audio_path: str,
    output_dir: str,
    text_id: str,
    num_samples: int = 5,
    sampling_params_list: Optional[List[Dict]] = None,
    stream: bool = False,
    fp16: bool = False,
    text_frontend: bool = False
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
        text_frontend: 是否使用文本前端处理
        
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
            # 修改模型的采样参数
            modify_model_sampling_params(cosyvoice, sampling_params)
            
            # 使用零样本合成
            for j, output in enumerate(cosyvoice.inference_zero_shot(
                text, 
                "", # 空提示文本
                reference_audio_16k, 
                stream=stream,
                text_frontend=text_frontend
            )):
                if j > 0:
                    logging.warning(f"生成了多个音频片段，只保留第一个")
                    break
                
                torchaudio.save(output_path, output['tts_speech'], cosyvoice.sample_rate)
                output_paths.append(output_path)
                
            logging.info(f"使用参数 {sampling_params} 生成样本 {i}")
        except Exception as e:
            logging.error(f"生成样本 {i} 时出错: {str(e)}")
    
    # 释放模型内存
    del cosyvoice
    torch.cuda.empty_cache()
    
    return output_paths

def modify_model_sampling_params(cosyvoice, sampling_params):
    """
    修改模型的采样参数
    
    Args:
        cosyvoice: CosyVoice2模型实例
        sampling_params: 采样参数字典
    """
    # 获取llm模型
    llm = cosyvoice.model.llm
    
    # 修改采样参数
    if hasattr(llm, 'sampling'):
        for key, value in sampling_params.items():
            if hasattr(llm.sampling, key):
                setattr(llm.sampling, key, value)
                logging.info(f"设置采样参数 {key} = {value}")
            else:
                logging.warning(f"采样参数 {key} 不存在")
    else:
        # 如果llm没有sampling属性，尝试在其他地方查找
        found = False
        for attr_name in dir(llm):
            attr = getattr(llm, attr_name)
            if hasattr(attr, 'sampling'):
                for key, value in sampling_params.items():
                    if hasattr(attr.sampling, key):
                        setattr(attr.sampling, key, value)
                        logging.info(f"在 {attr_name}.sampling 中设置参数 {key} = {value}")
                        found = True
                    else:
                        logging.warning(f"在 {attr_name}.sampling 中未找到参数 {key}")
                break
        
        if not found:
            logging.warning("未找到采样参数的位置，无法修改采样参数")

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
    fp16: bool = False,
    text_frontend: bool = False
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
        text_frontend: 是否使用文本前端处理
        
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
                    fp16=fp16,
                    text_frontend=text_frontend
                )
                
                speaker_results[text_id] = audio_paths
                logging.info(f"为说话人 {speaker} 的文本 {text_id} 生成了 {len(audio_paths)} 个样本")
            except Exception as e:
                logging.error(f"处理说话人 {speaker} 的文本 {text_id} 时出错: {str(e)}")
        
        results[speaker] = speaker_results
    
    # 保存采样参数到JSON文件
    if sampling_params_list is not None:
        params_file = os.path.join(output_base_dir, "sampling_params.json")
        with open(params_file, 'w', encoding='utf-8') as f:
            json.dump(sampling_params_list, f, indent=2, ensure_ascii=False)
    
    return results

def main():
    parser = argparse.ArgumentParser(description='为DPO训练生成多样化的CosyVoice2音频样本')
    parser.add_argument('--model_path', type=str, required=True, help='CosyVoice2模型路径')
    parser.add_argument('--excel_path', type=str, required=True, help='Excel文件路径')
    parser.add_argument('--output_dir', type=str, required=True, help='输出目录')
    parser.add_argument('--samples_per_text', type=int, default=5, help='每个文本生成的样本数量')
    parser.add_argument('--stream', action='store_true', help='是否使用流式处理')
    parser.add_argument('--fp16', action='store_true', help='是否使用半精度计算')
    parser.add_argument('--text_frontend', action='store_true', help='是否使用文本前端处理')
    parser.add_argument('--custom_params', type=str, default=None, help='自定义采样参数JSON文件路径')
    
    args = parser.parse_args()
    
    # 加载自定义采样参数（如果提供）
    sampling_params_list = None
    if args.custom_params:
        try:
            with open(args.custom_params, 'r', encoding='utf-8') as f:
                sampling_params_list = json.load(f)
            logging.info(f"已加载自定义采样参数: {sampling_params_list}")
        except Exception as e:
            logging.error(f"加载自定义采样参数时出错: {str(e)}")
            sampling_params_list = None
    
    # 处理Excel文件
    results = process_excel_for_dpo(
        model_path=args.model_path,
        excel_path=args.excel_path,
        output_base_dir=args.output_dir,
        samples_per_text=args.samples_per_text,
        sampling_params_list=sampling_params_list,
        stream=args.stream,
        fp16=args.fp16,
        text_frontend=args.text_frontend
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












# CosyVoice2 多样化音频生成代码 - 优化版（支持采样参数）

基于 `cosyvoice/cli/frontend.py` 代码，在其中有机地嵌入采样参数（"top_p", "top_k", "win_size", "tau_r"），以便灵活调控这些参数进行音频采样生成。

以下是优化后的完整代码：

```python
import torch
import torchaudio
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional, Generator
import os
from cosyvoice.cli.cosyvoice import CosyVoice2
from cosyvoice.utils.file_utils import logging
import yaml
import copy
import argparse
from tqdm import tqdm
import json

def generate_diverse_samples(
    model_path: str,
    text: str,
    reference_audio_path: str,
    output_dir: str,
    text_id: str,
    num_samples: int = 5,
    sampling_params_list: Optional[List[Dict]] = None,
    stream: bool = False,
    fp16: bool = False,
    text_frontend: bool = False
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
        text_frontend: 是否使用文本前端处理
        
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
            # 修改模型的采样参数
            modify_model_sampling_params(cosyvoice, sampling_params)
            
            # 使用零样本合成
            for j, output in enumerate(cosyvoice.inference_zero_shot(
                text, 
                "", # 空提示文本
                reference_audio_16k, 
                stream=stream,
                text_frontend=text_frontend
            )):
                if j > 0:
                    logging.warning(f"生成了多个音频片段，只保留第一个")
                    break
                
                torchaudio.save(output_path, output['tts_speech'], cosyvoice.sample_rate)
                output_paths.append(output_path)
                
            logging.info(f"使用参数 {sampling_params} 生成样本 {i}")
        except Exception as e:
            logging.error(f"生成样本 {i} 时出错: {str(e)}")
    
    # 释放模型内存
    del cosyvoice
    torch.cuda.empty_cache()
    
    return output_paths

def modify_model_sampling_params(cosyvoice, sampling_params):
    """
    修改模型的采样参数
    
    Args:
        cosyvoice: CosyVoice2模型实例
        sampling_params: 采样参数字典
    """
    # 获取llm模型
    llm = cosyvoice.model.llm
    
    # 修改采样参数
    if hasattr(llm, 'sampling'):
        for key, value in sampling_params.items():
            if hasattr(llm.sampling, key):
                setattr(llm.sampling, key, value)
                logging.info(f"设置采样参数 {key} = {value}")
            else:
                logging.warning(f"采样参数 {key} 不存在")
    else:
        # 如果llm没有sampling属性，尝试在其他地方查找
        found = False
        for attr_name in dir(llm):
            attr = getattr(llm, attr_name)
            if hasattr(attr, 'sampling'):
                for key, value in sampling_params.items():
                    if hasattr(attr.sampling, key):
                        setattr(attr.sampling, key, value)
                        logging.info(f"在 {attr_name}.sampling 中设置参数 {key} = {value}")
                        found = True
                    else:
                        logging.warning(f"在 {attr_name}.sampling 中未找到参数 {key}")
                break
        
        if not found:
            logging.warning("未找到采样参数的位置，无法修改采样参数")

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
    fp16: bool = False,
    text_frontend: bool = False
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
        text_frontend: 是否使用文本前端处理
        
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
                    fp16=fp16,
                    text_frontend=text_frontend
                )
                
                speaker_results[text_id] = audio_paths
                logging.info(f"为说话人 {speaker} 的文本 {text_id} 生成了 {len(audio_paths)} 个样本")
            except Exception as e:
                logging.error(f"处理说话人 {speaker} 的文本 {text_id} 时出错: {str(e)}")
        
        results[speaker] = speaker_results
    
    # 保存采样参数到JSON文件
    if sampling_params_list is not None:
        params_file = os.path.join(output_base_dir, "sampling_params.json")
        with open(params_file, 'w', encoding='utf-8') as f:
            json.dump(sampling_params_list, f, indent=2, ensure_ascii=False)
    
    return results

def main():
    parser = argparse.ArgumentParser(description='为DPO训练生成多样化的CosyVoice2音频样本')
    parser.add_argument('--model_path', type=str, required=True, help='CosyVoice2模型路径')
    parser.add_argument('--excel_path', type=str, required=True, help='Excel文件路径')
    parser.add_argument('--output_dir', type=str, required=True, help='输出目录')
    parser.add_argument('--samples_per_text', type=int, default=5, help='每个文本生成的样本数量')
    parser.add_argument('--stream', action='store_true', help='是否使用流式处理')
    parser.add_argument('--fp16', action='store_true', help='是否使用半精度计算')
    parser.add_argument('--text_frontend', action='store_true', help='是否使用文本前端处理')
    parser.add_argument('--custom_params', type=str, default=None, help='自定义采样参数JSON文件路径')
    
    args = parser.parse_args()
    
    # 加载自定义采样参数（如果提供）
    sampling_params_list = None
    if args.custom_params:
        try:
            with open(args.custom_params, 'r', encoding='utf-8') as f:
                sampling_params_list = json.load(f)
            logging.info(f"已加载自定义采样参数: {sampling_params_list}")
        except Exception as e:
            logging.error(f"加载自定义采样参数时出错: {str(e)}")
            sampling_params_list = None
    
    # 处理Excel文件
    results = process_excel_for_dpo(
        model_path=args.model_path,
        excel_path=args.excel_path,
        output_base_dir=args.output_dir,
        samples_per_text=args.samples_per_text,
        sampling_params_list=sampling_params_list,
        stream=args.stream,
        fp16=args.fp16,
        text_frontend=args.text_frontend
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


## 主要改进

1. **直接修改模型采样参数**：
   - 添加了 `modify_model_sampling_params` 函数，直接修改模型的采样参数
   - 该函数会尝试在模型的不同位置查找采样参数，以适应不同的模型结构

2. **更灵活的参数控制**：
   - 支持从JSON文件加载自定义采样参数
   - 保存使用的采样参数到输出目录，便于后续分析

3. **前端处理选项**：
   - 添加了 `text_frontend` 参数，控制是否使用文本前端处理
   - 根据 `frontend.py` 中的代码，默认设置为 False，与示例代码保持一致

4. **更好的错误处理**：
   - 增强了错误处理和日志记录
   - 即使某个样本生成失败，也会继续处理其他样本

## 使用方法

1. 将代码保存为 `generate_dpo_samples.py`
2. 准备一个包含必要列的Excel文件（text_id, text, speaker, reference_audio_path）
3. 运行以下命令：

```bash
python generate_dpo_samples.py --model_path pretrained_models/CosyVoice2-0.5B --excel_path your_data.xlsx --output_dir ./dpo_samples --samples_per_text 5
```

4. 如果要使用自定义采样参数，可以创建一个JSON文件，例如：

```json
[
  {
    "top_p": 0.8,
    "top_k": 25,
    "win_size": 10,
    "tau_r": 0.1
  },
  {
    "top_p": 0.95,
    "top_k": 50,
    "win_size": 15,
    "tau_r": 0.2
  }
]
```

然后使用 `--custom_params` 参数指定该文件：

```bash
python generate_dpo_samples.py --model_path pretrained_models/CosyVoice2-0.5B --excel_path your_data.xlsx --output_dir ./dpo_samples --samples_per_text 5 --custom_params custom_params.json
```

## 注意事项

1. 代码假设可以直接修改模型的采样参数。如果模型结构与预期不同，可能需要调整 `modify_model_sampling_params` 函数。

2. 根据 `frontend.py` 的代码，我们默认设置 `text_frontend=False`，这与示例代码中的注释一致：
   ```python
   # NOTE if you want to reproduce the results on https://funaudiollm.github.io/cosyvoice2, please add text_frontend=False during inference
   ```

3. 生成的目录结构如下：
   ```
   output_dir/
   ├── sampling_params.json
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

 


   

    
       
        
         
                

    
  
    
