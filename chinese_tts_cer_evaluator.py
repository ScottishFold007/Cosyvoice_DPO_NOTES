import os
import numpy as np
import tempfile
import soundfile as sf
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from pypinyin import lazy_pinyin
import re
import jieba
import Levenshtein

class FunASRCERCalculator:
    """基于FunASR的中文TTS字符错误率评估器"""
    
    def __init__(self, model_name="damo/speech_paraformer-large_asr_nat-zh-cn"):
        """
        初始化CER计算器
        Args:
            model_name: FunASR模型名称
        """
        # 初始化FunASR的ASR管道
        self.asr_pipeline = pipeline(
            task=Tasks.auto_speech_recognition,
            model=model_name,
            model_revision='v1.0.0'
        )
        
        # 临时文件存储目录
        self.temp_dir = tempfile.mkdtemp(prefix="cer_eval_")
        
    def preprocess_text(self, text, remove_punctuation=True, convert_to_lower=True):
        """
        文本预处理
        
        Args:
            text: 输入文本
            remove_punctuation: 是否移除标点符号
            convert_to_lower: 是否转换为小写
        
        Returns:
            处理后的文本
        """
        # 移除空白字符
        text = re.sub(r'\s+', '', text)
        
        if remove_punctuation:
            # 移除标点符号
            punctuation = r'[^\w\s]'
            text = re.sub(punctuation, '', text)
            
        if convert_to_lower:
            # 转换为小写
            text = text.lower()
            
        return text
    
    def compute_cer(self, audio, sr, reference_text, 
                    use_pinyin=False, 
                    detailed_analysis=False):
        """
        计算音频与参考文本的字符错误率
        
        Args:
            audio: 音频数据（numpy数组）
            sr: 采样率
            reference_text: 参考文本
            use_pinyin: 是否使用拼音进行比较（解决多音字问题）
            detailed_analysis: 是否返回详细错误分析
            
        Returns:
            CER值和详细分析（如果requested）
        """
        # 保存临时音频文件
        temp_file = os.path.join(self.temp_dir, "temp_audio.wav")
        sf.write(temp_file, audio, sr)
        
        # 使用FunASR进行语音识别
        asr_result = self.asr_pipeline(temp_file)
        transcribed_text = asr_result["text"]
        
        # 预处理参考文本和识别文本
        ref_text = self.preprocess_text(reference_text)
        hyp_text = self.preprocess_text(transcribed_text)
        
        if use_pinyin:
            # 转换为拼音进行比较（解决多音字问题）
            ref_pinyin = ' '.join(lazy_pinyin(ref_text))
            hyp_pinyin = ' '.join(lazy_pinyin(hyp_text))
            
            # 计算拼音CER
            distance = Levenshtein.distance(ref_pinyin, hyp_pinyin)
            cer = distance / max(len(ref_pinyin), 1)
            
            if detailed_analysis:
                # 计算拼音的插入、删除、替换错误
                ops = Levenshtein.editops(ref_pinyin, hyp_pinyin)
                insertions = sum(1 for op in ops if op[0] == 'insert')
                deletions = sum(1 for op in ops if op[0] == 'delete')
                substitutions = sum(1 for op in ops if op[0] == 'replace')
                
                pinyin_analysis = {
                    "insertions": insertions,
                    "deletions": deletions,
                    "substitutions": substitutions
                }
                
                return {
                    "cer": cer,
                    "ref_text": ref_text,
                    "hyp_text": hyp_text,
                    "ref_pinyin": ref_pinyin,
                    "hyp_pinyin": hyp_pinyin,
                    "pinyin_analysis": pinyin_analysis
                }
        else:
            # 直接计算字符CER
            distance = Levenshtein.distance(ref_text, hyp_text)
            cer = distance / max(len(ref_text), 1)
            
            if detailed_analysis:
                # 计算字符的插入、删除、替换错误
                ops = Levenshtein.editops(ref_text, hyp_text)
                insertions = sum(1 for op in ops if op[0] == 'insert')
                deletions = sum(1 for op in ops if op[0] == 'delete')
                substitutions = sum(1 for op in ops if op[0] == 'replace')
                
                # 中文分词后的粒度错误分析
                ref_words = list(jieba.cut(ref_text))
                hyp_words = list(jieba.cut(hyp_text))
                
                # 计算分词级别的编辑距离
                word_ops = Levenshtein.editops(''.join(ref_words), ''.join(hyp_words))
                word_errors = len(word_ops)
                word_error_rate = word_errors / max(len(ref_words), 1)
                
                return {
                    "cer": cer,
                    "ref_text": ref_text,
                    "hyp_text": hyp_text,
                    "char_level": {
                        "insertions": insertions,
                        "deletions": deletions,
                        "substitutions": substitutions,
                        "total_errors": distance,
                        "ref_length": len(ref_text)
                    },
                    "word_level": {
                        "word_error_rate": word_error_rate,
                        "ref_words": ref_words,
                        "hyp_words": hyp_words
                    }
                }
        
        # 清理临时文件
        if os.path.exists(temp_file):
            os.remove(temp_file)
            
        return cer
    
    def compute_batch_cer(self, audio_list, sample_rates, reference_texts):
        """计算批量音频的CER"""
        results = []
        
        for audio, sr, ref_text in zip(audio_list, sample_rates, reference_texts):
            cer = self.compute_cer(audio, sr, ref_text)
            results.append(cer)
            
        return results
    
    def analyze_cer_by_text_features(self, audio_list, sample_rates, reference_texts):
        """按文本特征分析CER表现"""
        # 1. 按长度分组
        length_groups = {
            "短文本(<=10字)": [],
            "中等文本(11-30字)": [],
            "长文本(>30字)": []
        }
        
        # 2. 按内容类型分组（需自定义分类逻辑）
        content_groups = {
            "问句": [],
            "陈述句": [],
            "感叹句": [],
            "对话": []
        }
        
        for audio, sr, text in zip(audio_list, sample_rates, reference_texts):
            cer = self.compute_cer(audio, sr, text)
            
            # 按长度分组
            if len(text) <= 10:
                length_groups["短文本(<=10字)"].append(cer)
            elif len(text) <= 30:
                length_groups["中等文本(11-30字)"].append(cer)
            else:
                length_groups["长文本(>30字)"].append(cer)
                
            # 按内容类型分组
            if "?" in text or "？" in text:
                content_groups["问句"].append(cer)
            elif "!" in text or "！" in text:
                content_groups["感叹句"].append(cer)
            elif re.search(r'[""].*[""]', text):
                content_groups["对话"].append(cer)
            else:
                content_groups["陈述句"].append(cer)
        
        # 计算各组的平均CER
        length_analysis = {k: np.mean(v) if v else None for k, v in length_groups.items()}
        content_analysis = {k: np.mean(v) if v else None for k, v in content_groups.items()}
        
        return {
            "length_analysis": length_analysis,
            "content_analysis": content_analysis
        }
    
    def analyze_error_patterns(self, audio, sr, reference_text):
        """分析错误模式"""
        # 使用详细分析模式获取错误信息
        result = self.compute_cer(audio, sr, reference_text, detailed_analysis=True)
        
        if isinstance(result, dict) and 'hyp_text' in result:
            ref_text = result['ref_text']
            hyp_text = result['hyp_text']
            
            # 分析常见错误模式
            # 1. 相似发音字符混淆
            similar_sound_errors = self._find_similar_sound_errors(ref_text, hyp_text)
            
            # 2. 分析多音字错误
            polyphonic_errors = self._find_polyphonic_errors(ref_text, hyp_text)
            
            # 3. 分析连续错误
            consecutive_errors = self._find_consecutive_errors(ref_text, hyp_text)
            
            # 4. 声调错误(适用于中文)
            tone_errors = self._find_tone_errors(ref_text, hyp_text)
            
            return {
                "cer": result['cer'],
                "error_patterns": {
                    "similar_sound_errors": similar_sound_errors,
                    "polyphonic_errors": polyphonic_errors,
                    "consecutive_errors": consecutive_errors,
                    "tone_errors": tone_errors
                }
            }
        
        return None
    
    def _find_similar_sound_errors(self, ref_text, hyp_text):
        """查找相似发音错误"""
        # 这需要根据中文发音相似字库实现
        # 简化版：将两者转为拼音，找出拼音相同但字不同的位置
        ref_pinyin = lazy_pinyin(ref_text)
        hyp_pinyin = lazy_pinyin(hyp_text)
        
        similar_errors = []
        min_len = min(len(ref_pinyin), len(hyp_pinyin))
        
        for i in range(min_len):
            if i < len(ref_text) and i < len(hyp_text):
                if ref_pinyin[i] == hyp_pinyin[i] and ref_text[i] != hyp_text[i]:
                    similar_errors.append({
                        "position": i,
                        "ref_char": ref_text[i],
                        "hyp_char": hyp_text[i],
                        "pinyin": ref_pinyin[i]
                    })
        
        return similar_errors
    
    def _find_polyphonic_errors(self, ref_text, hyp_text):
        """查找多音字错误"""
        # 需要多音字词典实现
        # 简化版：通过前后文判断
        polyphonic_errors = []
        # 实际实现中需使用多音字词典
        return polyphonic_errors
    
    def _find_consecutive_errors(self, ref_text, hyp_text):
        """查找连续错误"""
        ops = Levenshtein.editops(ref_text, hyp_text)
        
        # 按位置排序并查找连续错误
        consecutive_groups = []
        current_group = []
        
        for op in sorted(ops, key=lambda x: x[1]):
            if not current_group or op[1] == current_group[-1][1] + 1:
                current_group.append(op)
            else:
                if len(current_group) > 1:
                    consecutive_groups.append(current_group)
                current_group = [op]
                
        if len(current_group) > 1:
            consecutive_groups.append(current_group)
            
        # 提取连续错误信息
        consecutive_errors = []
        for group in consecutive_groups:
            start_pos = group[0][1]
            end_pos = group[-1][1]
            ref_segment = ref_text[start_pos:end_pos+1]
            
            # 提取假设文本对应段落
            # 注意：这里需要考虑插入和删除操作对索引的影响
            # 简化处理
            hyp_segment = ""
            for op in group:
                if op[0] != 'delete':  # 不是删除操作
                    hyp_idx = op[2]
                    if hyp_idx < len(hyp_text):
                        hyp_segment += hyp_text[hyp_idx]
            
            consecutive_errors.append({
                "start_pos": start_pos,
                "end_pos": end_pos,
                "ref_segment": ref_segment,
                "hyp_segment": hyp_segment,
                "error_length": len(group)
            })
            
        return consecutive_errors
    
    def _find_tone_errors(self, ref_text, hyp_text):
        """查找声调错误"""
        # 提取带声调的拼音
        ref_pinyin_with_tone = lazy_pinyin(ref_text, style=pypinyin.TONE)
        hyp_pinyin_with_tone = lazy_pinyin(hyp_text, style=pypinyin.TONE)
        
        # 提取不带声调的拼音
        ref_pinyin = lazy_pinyin(ref_text)
        hyp_pinyin = lazy_pinyin(hyp_text)
        
        tone_errors = []
        min_len = min(len(ref_pinyin), len(hyp_pinyin))
        
        for i in range(min_len):
            # 检查拼音相同但声调不同的情况
            if ref_pinyin[i] == hyp_pinyin[i] and ref_pinyin_with_tone[i] != hyp_pinyin_with_tone[i]:
                tone_errors.append({
                    "position": i,
                    "ref_char": ref_text[i] if i < len(ref_text) else "",
                    "hyp_char": hyp_text[i] if i < len(hyp_text) else "",
                    "ref_pinyin": ref_pinyin_with_tone[i],
                    "hyp_pinyin": hyp_pinyin_with_tone[i]
                })
                
        return tone_errors

