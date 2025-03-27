<think>
我需要具体解释如何实现和计算这些主观评估指标。这些都是语音合成评估中的指标，但可能不像PESQ或CER这样直接有标准算法或实现。我会详细解释每个维度的计算方法和具体实现建议。

首先，我来详细解释这些维度的计算方法：
</think>

# 语音评估指标的具体计算方法

下面我将详细解释这些看似主观的指标如何通过算法实现客观量化评估：

## 1. 韵律与表现力指标的计算方法

### 韵律自然度评分：
```python
def calculate_prosody_naturalness(audio, sr):
    # 提取F0（基频）轨迹
    f0, voiced_flag, voiced_probs = librosa.pyin(audio, 
                                                fmin=librosa.note_to_hz('C2'), 
                                                fmax=librosa.note_to_hz('C7'),
                                                sr=sr)
    
    # 1. 计算F0连续性 - 自然语音的F0轨迹应当平滑过渡
    f0_cleaned = f0[~np.isnan(f0)]
    if len(f0_cleaned) < 2:
        return 0.0
    
    f0_diff = np.diff(f0_cleaned)
    f0_continuity = 1.0 - min(1.0, np.mean(np.abs(f0_diff)) / 50.0)
    
    # 2. 计算能量变化自然度
    rms = librosa.feature.rms(y=audio)[0]
    rms_diff = np.diff(rms)
    energy_naturalness = 1.0 - min(1.0, np.std(rms_diff) / np.mean(rms) / 0.5)
    
    # 3. 分析停顿分布
    silence_threshold = 0.02
    is_silence = rms < silence_threshold
    silence_positions = np.where(np.diff(is_silence.astype(int)) != 0)[0]
    
    # 自然语音通常停顿应该遵循某种模式，不会过于频繁或稀疏
    if len(silence_positions) < 2:
        pause_score = 0.5  # 几乎没有停顿变化
    else:
        pause_intervals = np.diff(silence_positions)
        pause_regularity = 1.0 - min(1.0, np.std(pause_intervals) / np.mean(pause_intervals))
        pause_score = pause_regularity
    
    # 综合得分(0-5分制)
    prosody_naturalness = 5.0 * (0.4 * f0_continuity + 0.4 * energy_naturalness + 0.2 * pause_score)
    return prosody_naturalness
```

### 表达多样性指数：
```python
def calculate_expression_diversity(audio, sr):
    # 提取F0和能量
    f0, voiced_flag, voiced_probs = librosa.pyin(audio, sr=sr)
    rms = librosa.feature.rms(y=audio)[0]
    
    # 1. 计算F0变化范围与标准差
    f0_cleaned = f0[~np.isnan(f0)]
    if len(f0_cleaned) < 2:
        return 0.0
        
    f0_range = (np.max(f0_cleaned) - np.min(f0_cleaned)) / np.mean(f0_cleaned)
    f0_std = np.std(f0_cleaned) / np.mean(f0_cleaned)
    
    # 人类自然语音通常有一定的F0变化，但不会过于夸张
    # 正常范围约为0.2-0.6的标准差
    f0_diversity = min(1.0, f0_std / 0.3)
    
    # 2. 能量变化
    energy_range = (np.max(rms) - np.min(rms)) / (np.mean(rms) + 1e-8)
    energy_std = np.std(rms) / (np.mean(rms) + 1e-8)
    
    # 自然语音的能量变化适中
    energy_diversity = min(1.0, energy_std / 0.4)
    
    # 综合得分(0-5分制)
    diversity_score = 5.0 * (0.5 * f0_diversity + 0.5 * energy_diversity)
    return diversity_score
```

### 节奏感评分：
```python
def calculate_rhythm_score(audio, sr, text=None):
    # 提取音节边界 (可以使用预训练模型如Montreal Forced Aligner)
    # 这里用简化方法：利用能量变化检测
    rms = librosa.feature.rms(y=audio)[0]
    
    # 寻找能量峰值，这些通常对应音节核心
    peaks = librosa.util.peak_pick(rms, pre_max=3, post_max=3, pre_avg=3, post_avg=5, delta=0.1, wait=10)
    
    if len(peaks) < 2:
        return 0.0
    
    # 计算相邻峰值之间的间隔
    intervals = np.diff(peaks)
    
    # 1. 节奏规律性 - 评估语速变化的适当性
    # 过于机械的节奏会有非常规律的间隔，自然语音有适度变化
    rhythm_regularity = min(1.0, 0.5 * np.std(intervals) / np.mean(intervals))
    
    # 2. 停顿合理性
    silence_threshold = 0.02
    is_silence = rms < silence_threshold
    silence_ratio = np.mean(is_silence)
    
    # 停顿比例应适中，过多或过少都不自然
    pause_naturalness = 1.0 - 2.0 * abs(silence_ratio - 0.15)  # 假设15%为理想停顿比例
    pause_naturalness = max(0.0, pause_naturalness)
    
    # 3. 语速适当性
    if text:
        # 粗略计算语速：音节数/时长
        approx_syllables = max(1, len(text.split()) * 1.5)  # 英文粗略估计
        speech_duration = len(audio) / sr
        speech_rate = approx_syllables / speech_duration
        
        # 正常语速约为3-6音节/秒
        speech_rate_naturalness = 1.0 - min(1.0, abs(speech_rate - 4.5) / 3.0)
    else:
        speech_rate_naturalness = 0.5  # 无文本信息时默认中等得分
    
    # 综合得分(0-5分制)
    rhythm_score = 5.0 * (0.4 * rhythm_regularity + 0.3 * pause_naturalness + 0.3 * speech_rate_naturalness)
    return rhythm_score
```

## 2. 情感与表达指标的计算方法

### 情绪匹配度：
```python
def calculate_emotion_match(audio, text, sr, emotion_text_analyzer, emotion_audio_analyzer):
    # 1. 分析文本情绪
    text_emotion_result = emotion_text_analyzer(text)
    text_emotion_label = text_emotion_result[0]['label']
    text_emotion_score = text_emotion_result[0]['score']
    
    # 简化情绪类别映射 (根据具体情感分析模型调整)
    emotion_mapping = {
        'joy': 'happy',
        'sadness': 'sad',
        'anger': 'angry',
        'fear': 'fearful',
        'surprise': 'surprised',
        'neutral': 'neutral'
    }
    
    # 映射标准化情绪标签
    mapped_text_emotion = emotion_mapping.get(text_emotion_label, 'neutral')
    
    # 2. 分析音频情绪
    # 先将音频保存为临时文件(语音情感识别模型通常需要文件输入)
    temp_file = 'temp_audio.wav'
    sf.write(temp_file, audio, sr)
    
    audio_emotion_result = emotion_audio_analyzer(temp_file)
    audio_emotion_label = audio_emotion_result[0]['label']
    audio_emotion_score = audio_emotion_result[0]['score']
    
    # 映射音频情绪标签
    mapped_audio_emotion = emotion_mapping.get(audio_emotion_label, 'neutral')
    
    # 3. 计算匹配度
    if mapped_text_emotion == mapped_audio_emotion:
        # 完全匹配，取决于两者的置信度
        match_score = 5.0 * (text_emotion_score * audio_emotion_score)**0.5
    else:
        # 情绪空间距离计算 (需要预定义情绪向量空间)
        emotion_vectors = {
            'happy': [0.8, 0.2, -0.8, -0.5, 0.3],
            'sad': [-0.7, -0.3, -0.1, 0.5, -0.8],
            'angry': [0.1, -0.9, 0.8, -0.2, -0.5],
            'fearful': [-0.5, -0.7, 0.1, 0.8, -0.3],
            'surprised': [0.5, 0.1, 0.3, -0.1, 0.8],
            'neutral': [0.0, 0.0, 0.0, 0.0, 0.0]
        }
        
        if mapped_text_emotion in emotion_vectors and mapped_audio_emotion in emotion_vectors:
            # 计算余弦相似度
            vec1 = emotion_vectors[mapped_text_emotion]
            vec2 = emotion_vectors[mapped_audio_emotion]
            
            similarity = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
            similarity = (similarity + 1) / 2  # 归一化到0-1
            match_score = 5.0 * similarity
        else:
            match_score = 2.5  # 默认中等分
    
    # 删除临时文件
    if os.path.exists(temp_file):
        os.remove(temp_file)
        
    return match_score
```

### 情感强度评分：
```python
def calculate_emotion_intensity(audio, sr):
    # 1. 计算F0变化和范围
    f0, voiced_flag, voiced_probs = librosa.pyin(audio, sr=sr)
    f0_cleaned = f0[~np.isnan(f0)]
    
    if len(f0_cleaned) < 2:
        return 0.0
    
    f0_range = (np.max(f0_cleaned) - np.min(f0_cleaned)) / np.mean(f0_cleaned)
    f0_std = np.std(f0_cleaned) / np.mean(f0_cleaned)
    
    # 2. 计算能量变化
    rms = librosa.feature.rms(y=audio)[0]
    energy_range = (np.max(rms) - np.min(rms)) / (np.mean(rms) + 1e-8)
    energy_std = np.std(rms) / (np.mean(rms) + 1e-8)
    
    # 3. 计算语速变化
    # 利用零交叉率估计语速变化
    zcr = librosa.feature.zero_crossing_rate(audio)[0]
    zcr_std = np.std(zcr) / (np.mean(zcr) + 1e-8)
    
    # 情感强度综合评分(0-5分制)
    # 较大的变化通常表示更强的情感表达
    intensity_score = 5.0 * (0.4 * f0_range + 0.3 * energy_range + 0.3 * zcr_std)
    intensity_score = min(5.0, intensity_score)  # 上限5分
    
    return intensity_score
```

### 语气适当性评分：
```python
def calculate_tone_appropriateness(audio, text, sr, nlp_model=None):
    # 这个指标需要依赖文本内容分析和对应的音频表达
    # 1. 分析文本语境和意图
    if nlp_model:
        # 使用NLP模型分析文本语境
        doc = nlp_model(text)
        
        # 检测问句
        is_question = any(token.is_punct and token.text == '?' for token in doc)
        
        # 检测情感词汇
        sentiment_words = [token.text for token in doc if token._.sentiment != 0]
        sentiment_strength = sum(abs(token._.sentiment) for token in doc) / max(1, len(doc))
    else:
        # 简化版：基于标点和关键词
        is_question = '?' in text
        sentiment_strength = 0.5  # 默认中等情感强度
    
    # 2. 分析音频语气特征
    # 提取F0轮廓
    f0, voiced_flag, voiced_probs = librosa.pyin(audio, sr=sr)
    f0_cleaned = f0[~np.isnan(f0)]
    
    if len(f0_cleaned) < 2:
        return 0.0
    
    # 问句通常结尾音高上升
    if is_question:
        # 检查末尾部分的F0是否上升
        end_portion = max(int(len(f0_cleaned) * 0.8), 0)  # 获取最后20%
        end_f0 = f0_cleaned[end_portion:]
        
        if len(end_f0) > 1:
            end_trend = (end_f0[-1] - end_f0[0]) / end_f0[0]
            question_tone_match = 1.0 if end_trend > 0.05 else max(0, 1.0 - abs(end_trend) / 0.05)
        else:
            question_tone_match = 0.5
    else:
        question_tone_match = 1.0  # 非问句不需要上升语调
    
    # 情感表达强度是否与文本匹配
    emotion_intensity = calculate_emotion_intensity(audio, sr)
    emotion_match = 1.0 - min(1.0, abs(emotion_intensity/5.0 - sentiment_strength)/0.5)
    
    # 语气适当性综合评分(0-5分制)
    tone_score = 5.0 * (0.6 * question_tone_match + 0.4 * emotion_match)
    
    return tone_score
```

## 3. 说话人特性指标的计算方法

### 说话人相似度：
```python
def calculate_speaker_similarity(audio, sr, target_embedding):
    """计算与目标说话人的相似度"""
    # 使用说话人识别/验证模型 (例如ResemblyzerVoiceEncoder)
    voice_encoder = VoiceEncoder()
    
    # 提取当前音频的说话人嵌入
    audio_embedding = voice_encoder.embed_utterance(audio)
    
    # 计算与目标说话人嵌入的余弦相似度
    similarity = np.dot(audio_embedding, target_embedding) / (
        np.linalg.norm(audio_embedding) * np.linalg.norm(target_embedding)
    )
    
    # 相似度转换为0-5分
    similarity_score = 5.0 * (similarity + 1) / 2
    
    return similarity_score
```

### 声音稳定性指数：
```python
def calculate_voice_stability(audio, sr):
    """评估声音特征的稳定性"""
    # 将音频分成多个段落
    segment_length = int(0.5 * sr)  # 0.5秒段落
    num_segments = len(audio) // segment_length
    
    if num_segments < 2:
        return 2.5  # 音频太短，给予中等分数
    
    voice_encoder = VoiceEncoder()
    segment_embeddings = []
    
    # 为每个段落提取嵌入
    for i in range(num_segments):
        segment = audio[i*segment_length:(i+1)*segment_length]
        if len(segment) < 0.1 * sr:  # 段落过短
            continue
        try:
            embedding = voice_encoder.embed_utterance(segment)
            segment_embeddings.append(embedding)
        except:
            continue
    
    if len(segment_embeddings) < 2:
        return 2.5
    
    # 计算段落间的嵌入相似度
    similarities = []
    for i in range(len(segment_embeddings)):
        for j in range(i+1, len(segment_embeddings)):
            sim = np.dot(segment_embeddings[i], segment_embeddings[j]) / (
                np.linalg.norm(segment_embeddings[i]) * np.linalg.norm(segment_embeddings[j])
            )
            similarities.append(sim)
    
    # 取平均相似度作为稳定性指标
    avg_similarity = np.mean(similarities)
    stability_score = 5.0 * (avg_similarity + 1) / 2
    
    return stability_score
```

### 音色自然度评分：
```python
def calculate_timbre_naturalness(audio, sr):
    """评估音色是否自然不做作"""
    # 这需要较为复杂的评估，这里提供一个简化方法
    
    # 1. 提取梅尔频谱特征
    melspec = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=80)
    log_melspec = librosa.power_to_db(melspec)
    
    # 2. 计算频谱特征的统计特性
    # 自然语音的频谱分布通常有一定规律
    spec_mean = np.mean(log_melspec, axis=1)
    spec_std = np.std(log_melspec, axis=1)
    
    # 计算频谱分布的平滑度 (采用谱熵)
    spec_entropy = np.mean([entropy(frame) for frame in log_melspec.T if np.sum(frame) != 0])
    
    # 计算频谱平滑度 - 相邻帧之间的变化
    spec_smoothness = np.mean(np.abs(np.diff(log_melspec, axis=1)))
    
    # 3. 计算谐波噪声比 (Harmonic-to-Noise Ratio)
    # 简化版：使用ZCR和频谱平坦度作为代理
    zcr = librosa.feature.zero_crossing_rate(audio)[0]
    spec_flatness = librosa.feature.spectral_flatness(y=audio)[0]
    
    # 自然语音有适当的谐波结构和噪声比例
    hnr_proxy = np.mean(spec_flatness) * 10  # 值越小越自然
    hnr_score = 1.0 - min(1.0, hnr_proxy)
    
    # 4. 评估共振峰分布 (需要formant分析)
    # 这里简化为频谱峰值的分布评估
    spec_peaks = np.mean([len(find_peaks(frame)[0]) for frame in log_melspec.T])
    resonance_score = min(1.0, spec_peaks / 5.0)  # 假设平均5个峰值为理想
    
    # 综合评分(0-5分制)
    timbre_score = 5.0 * (0.3 * hnr_score + 0.3 * resonance_score + 
                         0.2 * (1.0 - min(1.0, spec_smoothness/5.0)) + 
                         0.2 * (1.0 - min(1.0, abs(spec_entropy - 4.0)/2.0)))
    
    return timbre_score
```

## 4. 连贯性与流畅度指标的计算方法

### 流畅度评分：
```python
def calculate_fluency_score(audio, sr):
    """分析语句流畅度"""
    # 1. 提取能量轮廓
    rms = librosa.feature.rms(y=audio)[0]
    
    # 2. 检测能量突变点
    energy_changes = np.abs(np.diff(rms))
    
    # 流畅的语音能量变化应该平滑
    sudden_changes = np.where(energy_changes > 0.1 * np.max(rms))[0]
    smooth_ratio = 1.0 - len(sudden_changes) / len(rms)
    
    # 3. 分析短暂停顿
    silence_threshold = 0.03 * np.max(rms)
    is_silence = rms < silence_threshold
    
    # 检测短暂停顿 (被认为会影响流畅度)
    silence_regions = []
    in_silence = False
    start_idx = 0
    
    for i, silent in enumerate(is_silence):
        if silent and not in_silence:
            in_silence = True
            start_idx = i
        elif not silent and in_silence:
            in_silence = False
            duration = i - start_idx
            # 记录较短的停顿
            if 3 < duration < 20:  # 调整阈值以识别不自然的短停顿
                silence_regions.append((start_idx, i))
    
    # 过多短停顿降低流畅度
    fluency_from_pauses = 1.0 - min(1.0, len(silence_regions) / 10.0)
    
    # 综合评分(0-5分制)
    fluency_score = 5.0 * (0.7 * smooth_ratio + 0.3 * fluency_from_pauses)
    
    return fluency_score
```

### 连贯性指数：
```python
def calculate_coherence_index(audio, sr, text=None):
    """评估语音连贯性"""
    # 1. 提取能量和基频
    rms = librosa.feature.rms(y=audio)[0]
    f0, voiced_flag, voiced_probs = librosa.pyin(audio, sr=sr)
    
    # 2. 检测自然段落边界
    # 段落边界通常有较长停顿
    silence_threshold = 0.05 * np.max(rms)
    is_silence = rms < silence_threshold
    
    # 检测较长停顿 (可能是段落边界)
    pause_regions = []
    in_silence = False
    start_idx = 0
    
    for i, silent in enumerate(is_silence):
        if silent and not in_silence:
            in_silence = True
            start_idx = i
        elif not silent and in_silence:
            in_silence = False
            duration = i - start_idx
            # 记录较长的停顿
            if duration > 30:  # 调整阈值以识别段落停顿
                pause_regions.append((start_idx, i))
    
    # 3. 分析句子间过渡
    # 句子间应当有连贯的韵律特征变化
    coherence_markers = 0
    
    for i in range(len(pause_regions)-1):
        pre_pause_end = pause_regions[i][1]
        post_pause_start = pause_regions[i+1][0]
        
        # 检查停顿前后的语调连贯性
        if pre_pause_end < len(f0) - 10 and post_pause_start > 10:
            pre_f0 = f0[pre_pause_end-10:pre_pause_end]
            post_f0 = f0[post_pause_start:post_pause_start+10]
            
            pre_f0_clean = pre_f0[~np.isnan(pre_f0)]
            post_f0_clean = post_f0[~np.isnan(post_f0)]
            
            if len(pre_f0_clean) > 0 and len(post_f0_clean) > 0:
                # 计算停顿前后语调的平滑过渡程度
                pre_mean = np.mean(pre_f0_clean)
                post_mean = np.mean(post_f0_clean)
                
                # 语调变化不应过大但也不应完全相同
                pitch_change = abs(post_mean - pre_mean) / pre_mean
                if 0.05 < pitch_change < 0.3:
                    coherence_markers += 1
    
    # 计算连贯性得分
    if len(pause_regions) <= 1:
        transition_coherence = 0.9  # 几乎没有停顿，默认较高连贯性
    else:
        transition_coherence = coherence_markers / (len(pause_regions) - 1)
    
    # 4. 语速一致性
    # 使用零交叉率估计语速变化
    zcr = librosa.feature.zero_crossing_rate(y=audio)[0]
    zcr_segments = np.array_split(zcr, max(1, len(pause_regions) + 1))
    
    if len(zcr_segments) > 1:
        segment_means = [np.mean(seg) for seg in zcr_segments]
        speed_consistency = 1.0 - min(1.0, np.std(segment_means) / np.mean(segment_means) / 0.3)
    else:
        speed_consistency = 1.0
    
    # 综合评分(0-5分制)
    coherence_score = 5.0 * (0.6 * transition_coherence + 0.4 * speed_consistency)
    
    return coherence_score
```

### 音节边界清晰度：
```python
def calculate_syllable_boundary_clarity(audio, sr):
    """评估音节边界的清晰度"""
    # 1. 提取语音信号特征
    # 使用能量变化和频谱变化检测音节边界
    rms = librosa.feature.rms(y=audio)[0]
    spec_flux = librosa.onset.onset_strength(y=audio, sr=sr)
    
    # 2. 检测音节边界 - 使用能量+频谱通量的变化点
    # 联合onset检测
    onsets = librosa.onset.onset_detect(
        onset_envelope=spec_flux,
        sr=sr,
        units='samples',
        hop_length=512,
        backtrack=True
    )
    
    if len(onsets) < 2:
        return 2.5  # 默认中等分数
    
    # 3. 评估边界清晰度
    # 计算边界处的对比度：边界前后能量/频谱变化的幅度
    boundary_contrasts = []
    window_size = 5  # 边界前后窗口大小
    
    for onset in onsets:
        onset_idx = min(onset // 512, len(spec_flux) - 1)  # 转换为spec_flux索引
        
        if onset_idx < window_size or onset_idx >= len(spec_flux) - window_size:
            continue
            
        pre_flux = spec_flux[onset_idx - window_size:onset_idx]
        post_flux = spec_flux[onset_idx:onset_idx + window_size]
        
        # 计算边界对比度
        contrast = np.abs(np.mean(post_flux) - np.mean(pre_flux)) / (np.mean(spec_flux) + 1e-8)
        boundary_contrasts.append(contrast)
    
    if not boundary_contrasts:
        return 2.5
    
    # 计算平均边界对比度和一致性
    avg_contrast = np.mean(boundary_contrasts)
    contrast_consistency = 1.0 - min(1.0, np.std(boundary_contrasts) / (avg_contrast + 1e-8) / 0.5)
    
    # 边界清晰度得分(0-5分制)
    # 清晰的边界应该有足够的对比度，但也不应过于夸张或不一致
    clarity_score = 5.0 * (0.7 * min(1.0, avg_contrast / 0.3) + 0.3 * contrast_consistency)
    
    return clarity_score
```

## 实际应用中的综合评分系统

在实际应用中，这些算法可以整合到一个统一的评估系统中，每个维度有对应的权重：

```python
def comprehensive_evaluation(audio, text, sr, target_speaker=None, reference_audio=None):
    """综合评估系统"""
    # 初始化评分字典
    scores = {}
    
    # 1. 韵律与表现力评分 (30%)
    scores['prosody_naturalness'] = calculate_prosody_naturalness(audio, sr)
    scores['expression_diversity'] = calculate_expression_diversity(audio, sr)
    scores['rhythm_score'] = calculate_rhythm_score(audio, sr, text)
    
    prosody_score = 0.5 * scores['prosody_naturalness'] + 0.3 * scores['expression_diversity'] + 0.2 * scores['rhythm_score']
    scores['prosody_overall'] = prosody_score
    
    # 2. 情感与表达评分 (20%)
    # 加载情感分析模型
    from transformers import pipeline
    emotion_text_analyzer = pipeline("text-classification", model="joeddav/distilbert-base-uncased-go-emotions-student")
    emotion_audio_analyzer = pipeline("audio-classification", model="ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition")
    
    scores['emotion_match'] = calculate_emotion_match(audio, text, sr, emotion_text_analyzer, emotion_audio_analyzer)
    scores['emotion_intensity'] = calculate_emotion_intensity(audio, sr)
    scores['tone_appropriateness'] = calculate_tone_appropriateness(audio, text, sr)
    
    emotion_score = 0.5 * scores['emotion_match'] + 0.3 * scores['emotion_intensity'] + 0.2 * scores['tone_appropriateness']
    scores['emotion_overall'] = emotion_score
    
    # 3. 说话人特性评分 (20%)
    if target_speaker is not None:
        scores['speaker_similarity'] = calculate_speaker_similarity(audio, sr, target_speaker)
    else:
        scores['speaker_similarity'] = None
        
    scores['voice_stability'] = calculate_voice_stability(audio, sr)
    scores['timbre_naturalness'] = calculate_timbre_naturalness(audio, sr)
    
    if scores['speaker_similarity'] is not None:
        speaker_score = 0.5 * scores['speaker_similarity'] + 0.3 * scores['voice_stability'] + 0.2 * scores['timbre_naturalness']
    else:
        speaker_score = 0.6 * scores['voice_stability'] + 0.4 * scores['timbre_naturalness']
        
    scores['speaker_overall'] = speaker_score
    
    # 4. 连贯性与流畅度评分 (20%)
    scores['fluency'] = calculate_fluency_score(audio, sr)
    scores['coherence'] = calculate_coherence_index(audio, sr, text)
    scores['syllable_clarity'] = calculate_syllable_boundary_clarity(audio, sr)
    
    fluency_score = 0.4 * scores['fluency'] + 0.4 * scores['coherence'] + 0.2 * scores['syllable_clarity']
    scores['fluency_overall'] = fluency_score
    
    # 5. 文本准确性评分 (10%)
    # 使用ASR模型计算CER
    import whisper
    asr_model = whisper.load_model("base")
    
    # 保存临时文件
    import soundfile as sf
    temp_file = "temp_eval.wav"
    sf.write(temp_file, audio, sr)
    
    # 转写音频
    result = asr_model.transcribe(temp_file)
    transcribed_text = result["text"]
    
    # 计算CER
    import jiwer
    transformation = jiwer.Compose([
        jiwer.ToLowerCase(),
        jiwer.RemoveMultipleSpaces(),
        jiwer.Strip(),
        jiwer.RemovePunctuation()
    ])
    cer = jiwer.cer(
        transformation(text),
        transformation(transcribed_text)
    )
    
    scores['cer'] = cer
    # CER越低越好，转换为5分制评分
    text_accuracy_score = 5.0 * max(0, 1.0 - cer / 0.5)  # 0.5以上的CER得0分
    scores['text_accuracy'] = text_accuracy_score
    
    # 删除临时文件
    import os
    if os.path.exists(temp_file):
        os.remove(temp_file)
    
    # 计算总体评分 (加权平均)
    overall_score = (
        0.3 * prosody_score +
        0.2 * emotion_score +
        0.2 * speaker_score +
        0.2 * fluency_score +
        0.1 * text_accuracy_score
    )
    
    scores['overall'] = overall_score
    
    return scores
```

通过这个全面的评估系统，我们可以客观地量化语音合成的各个质量维度，为不同采样参数的语音生成结果提供有意义的对比依据。
