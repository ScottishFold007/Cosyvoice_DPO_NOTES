
# 语音合成全方位评估系统

梳理SpeechScore的质量评估维度，下面是一个系统性的语音质量评估框架，涵盖多个关键维度：

## 完整评估维度体系

### 1. 语音准确性与可懂度
| 维度 | 指标 | 评估方法 | 重要性 |
|------|------|----------|--------|
| **文本准确性** | CER (字符错误率) | 使用ASR模型转写音频，与原文本比较 | 非常高 |
| **可懂度** | STOI | 短时客观可懂度指数 | 高 |
| **清晰度** | CSIG | 信号失真平均意见分 | 高 |

### 2. 音质与自然度
| 维度 | 指标 | 评估方法 | 重要性 |
|------|------|----------|--------|
| **感知质量** | PESQ/NB_PESQ | 感知评估语音质量 | 高 |
| **整体质量** | DNSMOS (OVRL/P808_MOS) | 整体质量评分 | 高 |
| **背景清洁度** | CBAK, DNSMOS (BAK) | 背景侵入性测量 | 中 |
| **声学特性** | MCD, LSD | 梅尔倒谱失真、对数谱距离 | 中 |
| **信噪比** | FWSEGSNR, SNR | 频率加权分段信噪比 | 中 |

### 3. 韵律与表现力
| 维度 | 指标 | 评估方法 | 重要性 |
|------|------|----------|--------|
| **韵律自然度** | 韵律得分 | 分析F0轮廓、能量变化与停顿分布 | 高 |
| **多样性** | 表达多样性指数 | 分析音高/能量变化范围与标准差 | 中高 |
| **节奏感** | 节奏评分 | 音节持续时间分布与停顿模式分析 | 中 |

### 4. 情感与表达
| 维度 | 指标 | 评估方法 | 重要性 |
|------|------|----------|--------|
| **情绪匹配度** | 情绪一致性分数 | 文本情绪分析与音频情绪识别匹配度 | 高 |
| **情感强度** | 情感表达强度 | 测量情感表达的显著程度 | 中高 |
| **语气适当性** | 语气匹配分数 | 评估语气与文本语境的匹配程度 | 中高 |

### 5. 说话人特性
| 维度 | 指标 | 评估方法 | 重要性 |
|------|------|----------|--------|
| **说话人相似度** | 说话人嵌入距离 | 与目标说话人的声音特征相似度 | 高 |
| **声音稳定性** | 声音一致性指数 | 评估整段语音中声音特征的稳定性 | 中 |
| **音色自然度** | 音色自然度分数 | 评估音色是否自然不做作 | 中 |

### 6. 连贯性与流畅度
| 维度 | 指标 | 评估方法 | 重要性 |
|------|------|----------|--------|
| **流畅度** | 流畅度评分 | 分析语句间过渡的自然性 | 高 |
| **连贯性** | 连贯性指数 | 评估语音中断、停顿的合理性 | 高 |
| **音节边界** | 边界清晰度 | 评估音节边界的清晰与自然 | 中 |

## 实现方案

```python:comprehensive_speech_evaluator.py
import torch
import numpy as np
import librosa
import soundfile as sf
import tempfile
import os
from speechscore import SpeechScore
from resemblyzer import VoiceEncoder  # 说话人相似度
import whisper  # ASR转写
from transformers import pipeline  # 情感分析

class ComprehensiveSpeechEvaluator:
    """全方位语音评估系统"""
    
    def __init__(self, reference_dir=None, speaker_embedding=None):
        """
        Args:
            reference_dir: 参考语音目录
            speaker_embedding: 目标说话人嵌入
        """
        # SpeechScore评估器
        self.speech_score = SpeechScore([
            'SRMR', 'PESQ', 'NB_PESQ', 'STOI', 'SISDR', 
            'FWSEGSNR', 'LSD', 'DNSMOS', 'SNR', 'CSIG', 
            'CBAK', 'COVL', 'MCD'
        ])
        
        # 临时目录
        self.temp_dir = tempfile.mkdtemp(prefix="speech_eval_")
        self.reference_dir = reference_dir
        self.speaker_embedding = speaker_embedding
        
        # 加载ASR模型（用于CER计算）
        self.asr_model = whisper.load_model("base")
        
        # 加载情绪分析模型
        self.emotion_text_analyzer = pipeline("text-classification", 
                                         model="joeddav/distilbert-base-uncased-go-emotions-student")
        
        # 加载情绪识别模型（音频）
        self.emotion_audio_analyzer = pipeline("audio-classification", 
                                          model="ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition")
        
        # 加载说话人验证模型
        self.voice_encoder = VoiceEncoder()
        
        # 韵律分析工具
        self.prosody_analyzer = self._initialize_prosody_analyzer()
    
    def _initialize_prosody_analyzer(self):
        """初始化韵律分析工具"""
        # 这里可以使用专门的韵律分析库或自定义分析函数
        # 简化版本，返回分析函数
        def analyze_prosody(audio, sr):
            # 提取F0
            f0, voiced_flag, voiced_probs = librosa.pyin(
                audio, fmin=librosa.note_to_hz('C2'), 
                fmax=librosa.note_to_hz('C7'),
                sr=sr
            )
            
            # 计算韵律特征
            f0_mean = np.nanmean(f0) if not np.all(np.isnan(f0)) else 0
            f0_std = np.nanstd(f0) if not np.all(np.isnan(f0)) else 0
            f0_range = np.nanmax(f0) - np.nanmin(f0) if not np.all(np.isnan(f0)) else 0
            
            # 计算能量特征
            rms_energy = librosa.feature.rms(y=audio)[0]
            energy_mean = np.mean(rms_energy)
            energy_std = np.std(rms_energy)
            
            # 计算停顿特征
            silence_threshold = 0.02
            is_silence = rms_energy < silence_threshold
            silence_ratio = np.mean(is_silence)
            
            # 结合特征计算评分
            prosody_features = {
                'f0_mean': f0_mean,
                'f0_std': f0_std,
                'f0_range': f0_range,
                'energy_mean': energy_mean,
                'energy_std': energy_std,
                'silence_ratio': silence_ratio
            }
            
            # 计算韵律评分（简化版本）
            prosody_score = min(5.0, 2.5 + f0_std/50 + energy_std*10 - abs(silence_ratio-0.2)*5)
            
            return {
                'prosody_score': prosody_score,
                'pitch_variability': f0_std,
                'energy_variability': energy_std,
                'silence_distribution': silence_ratio,
                'features': prosody_features
            }
            
        return analyze_prosody
    
    def __del__(self):
        """清理临时文件"""
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def calculate_cer(self, audio_np, sr, reference_text):
        """计算字符错误率"""
        # 使用ASR模型转写
        result = self.asr_model.transcribe(audio_np, sampling_rate=sr)
        transcribed_text = result["text"].strip()
        
        # 计算CER
        from jiwer import wer
        
        # 预处理文本
        def preprocess_text(text):
            import re
            # 移除标点符号，转换为小写
            text = re.sub(r'[^\w\s]', '', text.lower())
            return text
        
        ref_processed = preprocess_text(reference_text)
        hyp_processed = preprocess_text(transcribed_text)
        
        # 计算字符级别错误率
        cer = wer(list(ref_processed), list(hyp_processed))
        
        return {
            'cer': cer,
            'transcribed_text': transcribed_text,
            'reference_text': reference_text
        }
    
    def analyze_emotion_match(self, audio_np, sr, text):
        """分析情绪匹配度"""
        # 保存临时文件用于情绪分析
        temp_path = os.path.join(self.temp_dir, f"temp_emotion_{hash(text)}.wav")
        sf.write(temp_path, audio_np, sr)
        
        # 文本情绪分析
        text_emotion = self.emotion_text_analyzer(text)
        text_emotion_label = text_emotion[0]['label']
        text_emotion_score = text_emotion[0]['score']
        
        # 音频情绪分析
        audio_emotion = self.emotion_audio_analyzer(temp_path)
        audio_emotion_label = audio_emotion[0]['label']
        audio_emotion_score = audio_emotion[0]['score']
        
        # 情绪映射（简化处理）
        emotion_groups = {
            'positive': ['joy', 'happiness', 'excited', 'admiration', 'amusement', 'approval'],
            'negative': ['sadness', 'anger', 'disgust', 'fear', 'disappointment', 'annoyance'],
            'neutral': ['neutral', 'calm', 'realization', 'confusion'],
            'surprised': ['surprise', 'amazement']
        }
        
        # 获取情绪组
        def get_emotion_group(emotion):
            for group, emotions in emotion_groups.items():
                if emotion in emotions:
                    return group
            return 'other'
        
        text_emotion_group = get_emotion_group(text_emotion_label)
        audio_emotion_group = get_emotion_group(audio_emotion_label)
        
        # 计算匹配分数
        if text_emotion_group == audio_emotion_group:
            match_score = 1.0
        elif (text_emotion_group == 'neutral' or audio_emotion_group == 'neutral'):
            match_score = 0.7
        else:
            match_score = 0.3
            
        # 加权考虑情绪强度
        emotion_strength = (text_emotion_score + audio_emotion_score) / 2
        emotion_match_score = 5 * match_score * emotion_strength
        
        return {
            'emotion_match_score': emotion_match_score,
            'text_emotion': text_emotion_label,
            'audio_emotion': audio_emotion_label,
            'text_emotion_group': text_emotion_group,
            'audio_emotion_group': audio_emotion_group,
            'text_emotion_confidence': text_emotion_score,
            'audio_emotion_confidence': audio_emotion_score
        }
    
    def calculate_speaker_similarity(self, audio_np, sr):
        """计算说话人相似度"""
        if self.speaker_embedding is None:
            return {'speaker_similarity': None}
        
        # 提取说话人嵌入
        audio_embedding = self.voice_encoder.embed_utterance(audio_np)
        
        # 计算余弦相似度
        similarity = np.dot(audio_embedding, self.speaker_embedding) / (
            np.linalg.norm(audio_embedding) * np.linalg.norm(self.speaker_embedding))
        
        # 转换为0-5分制
        similarity_score = 5 * (similarity + 1) / 2
        
        return {
            'speaker_similarity': similarity_score,
            'raw_similarity': similarity
        }
    
    def evaluate(self, speech_tensor, text, reference_path=None, sr=24000):
        """全方位评估语音样本"""
        # 转换为numpy数组
        speech_np = speech_tensor.squeeze(0).cpu().numpy()
        
        # 创建临时WAV文件
        temp_path = os.path.join(self.temp_dir, f"temp_{hash(text)}.wav")
        sf.write(temp_path, speech_np, sr)
        
        all_metrics = {}
        
        # 1. SpeechScore基础指标评估
        if reference_path:
            speech_scores = self.speech_score(
                test_path=temp_path,
                reference_path=reference_path,
                window=None,
                score_rate=sr,
                return_mean=False
            )
        else:
            speech_scores = self.speech_score(
                test_path=temp_path,
                window=None,
                score_rate=sr,
                return_mean=False
            )
        
        all_metrics.update(speech_scores)
        
        # 2. 计算CER
        cer_result = self.calculate_cer(speech_np, sr, text)
        all_metrics['CER'] = cer_result
        
        # 3. 情绪匹配度分析
        emotion_result = self.analyze_emotion_match(speech_np, sr, text)
        all_metrics['Emotion'] = emotion_result
        
        # 4. 韵律分析
        prosody_result = self.prosody_analyzer(speech_np, sr)
        all_metrics['Prosody'] = prosody_result
        
        # 5. 说话人相似度（如果有目标说话人嵌入）
        if self.speaker_embedding is not None:
            speaker_result = self.calculate_speaker_similarity(speech_np, sr)
            all_metrics['Speaker'] = speaker_result
        
        # 6. 计算综合评分
        overall_score = self._compute_overall_score(all_metrics)
        all_metrics['overall_score'] = overall_score
        
        return all_metrics
    
    def _compute_overall_score(self, metrics):
        """计算综合评分"""
        score_components = []
        weights = []
        
        # 1. 准确性与可懂度权重
        if 'CER' in metrics:
            cer_score = max(0, 5 * (1 - metrics['CER']['cer']))
            score_components.append(cer_score)
            weights.append(0.25)  # 高权重
            
        if 'STOI' in metrics:
            stoi_score = 5 * metrics['STOI']
            score_components.append(stoi_score)
            weights.append(0.10)
        
        # 2. 音质评分权重
        if 'PESQ' in metrics:
            pesq_score = min(5, metrics['PESQ'])
            score_components.append(pesq_score)
            weights.append(0.10)
            
        if 'DNSMOS' in metrics and 'OVRL' in metrics['DNSMOS']:
            dnsmos_score = metrics['DNSMOS']['OVRL']
            score_components.append(dnsmos_score)
            weights.append(0.10)
        
        # 3. 情绪匹配度权重
        if 'Emotion' in metrics:
            emotion_score = metrics['Emotion']['emotion_match_score']
            score_components.append(emotion_score)
            weights.append(0.15)  # 高权重
        
        # 4. 韵律评分权重
        if 'Prosody' in metrics:
            prosody_score = metrics['Prosody']['prosody_score']
            score_components.append(prosody_score)
            weights.append(0.15)  # 高权重
        
        # 5. 说话人相似度权重
        if 'Speaker' in metrics and metrics['Speaker']['speaker_similarity'] is not None:
            speaker_score = metrics['Speaker']['speaker_similarity']
            score_components.append(speaker_score)
            weights.append(0.15)  # 高权重
        
        # 计算加权平均分
        if len(score_components) > 0:
            weighted_sum = sum(s * w for s, w in zip(score_components, weights))
            weight_sum = sum(weights)
            overall_score = weighted_sum / weight_sum
        else:
            overall_score = 0.0
            
        return overall_score

    def evaluate_multiple_variants(self, variants, text, reference_path=None):
        """评估同一文本的多个语音变体"""
        scored_variants = []
        
        for variant in variants:
            # 评估单个变体
            scores = self.evaluate(
                speech_tensor=variant['speech'],
                text=text,
                reference_path=reference_path
            )
            
            scored_variants.append({
                'id': variant.get('id', f"variant_{len(scored_variants)}"),
                'speech': variant['speech'],
                'config': variant.get('config', {}),
                'scores': scores,
                'overall_score': scores['overall_score']
            })
        
        # 按整体分数排序
        scored_variants.sort(key=lambda x: x['overall_score'], reverse=True)
        
        return scored_variants
```

## 评估维度的解释与意义

### 1. 文本准确性与可懂度 (25%)
衡量生成语音能否正确、清晰地传达原始文本，这是最基础的要求。CER与STOI指标能有效评估这方面的表现。

### 2. 音质与自然度 (20%)
评估语音是否清晰、自然，无明显合成痕迹。PESQ、DNSMOS等指标可全面评估音质各方面。

### 3. 情绪匹配度 (15%)
评估语音表达的情绪是否与文本内容匹配，例如高兴、悲伤、愤怒等情绪是否得到适当表达。

### 4. 韵律与表现力 (15%)
评估语音的抑扬顿挫、节奏变化是否合理自然，避免单调乏味或过于戏剧化。

### 5. 说话人相似度 (15%)
评估生成语音是否保持目标说话人的声音特征，保持一致的音色和风格。

### 6. 连贯性与流畅度 (10%)
评估语音是否流畅自然，句子之间过渡是否合理，无不自然的停顿或断裂。

## 数据收集与分析示例

```python
def collect_quality_analysis(model, texts, style_configs, evaluator):
    """收集不同风格配置下的质量分析数据"""
    analysis_results = {}
    
    for style_name, config in style_configs.items():
        print(f"分析 '{style_name}' 风格...")
        style_scores = []
        
        # 更新模型采样参数
        model.update_sampling_params(**config)
        
        for text in texts[:10]:  # 使用部分文本做测试
            # 生成语音
            speech = next(model.tts(text, embedding))['tts_speech']
            
            # 评估语音
            scores = evaluator.evaluate(speech, text)
            
            # 记录结果
            style_scores.append({
                'text': text,
                'overall_score': scores['overall_score'],
                'cer': scores['CER']['cer'],
                'emotion_match': scores['Emotion']['emotion_match_score'],
                'prosody': scores['Prosody']['prosody_score'],
                'audio_quality': scores.get('PESQ', 0),
                'intelligibility': scores.get('STOI', 0)
            })
        
        # 计算平均分
        avg_scores = {
            'overall': np.mean([s['overall_score'] for s in style_scores]),
            'cer': np.mean([s['cer'] for s in style_scores]),
            'emotion': np.mean([s['emotion_match'] for s in style_scores]),
            'prosody': np.mean([s['prosody'] for s in style_scores]),
            'quality': np.mean([s['audio_quality'] for s in style_scores]),
            'intelligibility': np.mean([s['intelligibility'] for s in style_scores])
        }
        
        analysis_results[style_name] = {
            'config': config,
            'average_scores': avg_scores,
            'detailed_scores': style_scores
        }
    
    return analysis_results
```

## 参数对质量影响分析

通过系统性分析不同采样参数对各维度评分的影响，我们可以得出以下关系：

1. **top_p 参数**:
   - 较高值(0.9+): 增加多样性和表现力，但可能降低准确性
   - 较低值(0.7-): 提高准确性和稳定性，但可能降低表现力
   
2. **top_k 参数**:
   - 较高值(50+): 增加创意性和随机性，适合情感表达
   - 较低值(25-): 提高文本忠实度，适合准确朗读
   
3. **win_size 和 tau_r 参数**:
   - win_size越大，考虑的历史越长，有助于维持连贯性
   - tau_r越高，对重复的惩罚越强，增加表达多样性

## 最佳实践建议

1. **文档朗读场景**: 使用低top_p(0.7)和低top_k(10)配置，确保准确性
2. **情感表达场景**: 使用高top_p(0.9)和适中top_k(50)，增强情感表现
3. **戏剧/表演场景**: 使用高top_p(0.95+)和高top_k(100+)，最大化表现力
4. **正式新闻播报**: 使用中等top_p(0.8)和低top_k(25)，平衡准确性和表现力

 
