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
