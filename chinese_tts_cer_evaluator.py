import os
import numpy as np
import tempfile
import soundfile as sf
import json
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
import pypinyin
from pypinyin import lazy_pinyin, Style
import re
import jieba
import Levenshtein

class FunASRCERCalculator:
    """基于FunASR的中文TTS字符错误率评估器，支持多音字处理"""
    # 多音字字典地址：https://github.com/mapull/chinese-dictionary/blob/main/character/polyphone.json
    def __init__(self, model_name="damo/speech_paraformer-large_asr_nat-zh-cn", 
                 polyphone_path="polyphone.json"):
        """
        初始化CER计算器
        Args:
            model_name: FunASR模型名称
            polyphone_path: 多音字词典路径
        """
        # 初始化FunASR的ASR管道
        self.asr_pipeline = pipeline(
            task=Tasks.auto_speech_recognition,
            model=model_name,
            model_revision='v1.0.0'
        )
        
        # 临时文件存储目录
        self.temp_dir = tempfile.mkdtemp(prefix="cer_eval_")
        
        # 加载多音字字典
        self.polyphone_dict = self._load_polyphone_dict(polyphone_path)
        
    def _load_polyphone_dict(self, path):
        """
        加载多音字字典
        
        Args:
            path: 字典文件路径
            
        Returns:
            多音字映射字典
        """
        try:
            with open(path, 'r', encoding='utf-8') as f:
                polyphone_list = json.load(f)
                
            # 转换为更易用的格式：{'字符': ['拼音1', '拼音2', ...]}
            polyphone_dict = {}
            for item in polyphone_list:
                char = item.get('char')
                pinyins = item.get('pinyin', [])
                
                if char and pinyins:
                    # 去除音调标记，便于比较
                    normalized_pinyins = []
                    for p in pinyins:
                        # 去除音调并转小写
                        normalized_p = re.sub(r'[àáâãäåāǎ]', 'a', p)
                        normalized_p = re.sub(r'[èéêëēěẽ]', 'e', normalized_p)
                        normalized_p = re.sub(r'[ìíîïīǐĩ]', 'i', normalized_p)
                        normalized_p = re.sub(r'[òóôõöōǒ]', 'o', normalized_p)
                        normalized_p = re.sub(r'[ùúûüūǔũ]', 'u', normalized_p)
                        normalized_p = re.sub(r'[ǚǜǘǖü]', 'v', normalized_p)
                        normalized_pinyins.append(normalized_p.lower())
                    
                    polyphone_dict[char] = {
                        'original': pinyins,  # 带声调拼音
                        'normalized': normalized_pinyins  # 无声调拼音
                    }
            
            print(f"成功加载多音字字典，共{len(polyphone_dict)}个多音字")
            return polyphone_dict
        except Exception as e:
            print(f"加载多音字字典失败: {e}")
            return {}
    
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
                    consider_polyphones=True, 
                    detailed_analysis=False):
        """
        计算音频与参考文本的字符错误率
        
        Args:
            audio: 音频数据（numpy数组）
            sr: 采样率
            reference_text: 参考文本
            consider_polyphones: 是否考虑多音字
            detailed_analysis: 是否返回详细错误分析
            
        Returns:
            CER值和详细分析（如果requested）
        """
        # 保存音频到临时文件
        temp_wav_path = os.path.join(self.temp_dir, "temp_audio.wav")
        sf.write(temp_wav_path, audio, sr)
        
        # 使用FunASR转写音频
        asr_result = self.asr_pipeline(temp_wav_path)
        transcribed_text = asr_result['text']
        
        # 预处理参考文本和转写文本
        ref_text = self.preprocess_text(reference_text)
        hyp_text = self.preprocess_text(transcribed_text)
        
        # 基础CER计算
        distance = self._calculate_edit_distance(ref_text, hyp_text, consider_polyphones)
        ref_length = len(ref_text)
        cer = distance / ref_length if ref_length > 0 else 0.0
        
        if not detailed_analysis:
            return cer
        
        # 详细分析
        analysis_result = {
            'cer': cer,
            'ref_text': ref_text,
            'hyp_text': hyp_text,
            'audio_path': temp_wav_path,
            'char_level': self._analyze_char_errors(ref_text, hyp_text, consider_polyphones),
            'pinyin_level': self._analyze_pinyin_errors(ref_text, hyp_text, consider_polyphones),
            'word_level': self._analyze_word_errors(reference_text, transcribed_text)
        }
        
        # 添加错误模式分析
        analysis_result['error_patterns'] = self._analyze_error_patterns(ref_text, hyp_text)
        
        return analysis_result
    
    def _calculate_edit_distance(self, ref_text, hyp_text, consider_polyphones=True):
        """
        计算编辑距离，可选择考虑多音字
        
        Args:
            ref_text: 参考文本
            hyp_text: 转写文本
            consider_polyphones: 是否考虑多音字
            
        Returns:
            编辑距离
        """
        if not consider_polyphones:
            # 标准编辑距离计算
            return Levenshtein.distance(ref_text, hyp_text)
        
        # 考虑多音字情况下的编辑距离计算
        # 可使用动态规划实现自定义编辑距离
        m, n = len(ref_text), len(hyp_text)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        # 初始化边界
        for i in range(m + 1):
            dp[i][0] = i
        for j in range(n + 1):
            dp[0][j] = j
        
        # 动态规划填表
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                # 计算替换代价，考虑多音字
                substitution_cost = 1
                if ref_text[i-1] == hyp_text[j-1]:
                    substitution_cost = 0
                elif self._is_valid_polyphone_variant(ref_text[i-1], hyp_text[j-1]):
                    # 如果是有效的多音字变体，不计错误
                    substitution_cost = 0
                
                # 计算插入、删除和替换操作的代价
                dp[i][j] = min(
                    dp[i-1][j] + 1,          # 删除
                    dp[i][j-1] + 1,          # 插入
                    dp[i-1][j-1] + substitution_cost  # 替换
                )
        
        return dp[m][n]
    
    def _is_valid_polyphone_variant(self, char1, char2):
        """
        检查两个字符是否是相同的多音字的有效变体
        本例直接检查两个字符是否相同，多音字问题通过拼音处理
        
        Args:
            char1: 第一个字符
            char2: 第二个字符
            
        Returns:
            是否为有效变体
        """
        # 字符相同直接返回True
        if char1 == char2:
            return True
            
        # 目前仅检查完全相同的字符
        # 更复杂的多音字变体处理可以在未来实现
        return False
        
    def _analyze_char_errors(self, ref_text, hyp_text, consider_polyphones=True):
        """
        分析字符级别的错误
        
        Args:
            ref_text: 参考文本
            hyp_text: 转写文本
            consider_polyphones: 是否考虑多音字
            
        Returns:
            字符错误分析
        """
        # 获取编辑操作
        operations = Levenshtein.editops(ref_text, hyp_text)
        
        # 统计各类操作
        substitutions = [op for op in operations if op[0] == 'replace']
        insertions = [op for op in operations if op[0] == 'insert']
        deletions = [op for op in operations if op[0] == 'delete']
        
        # 过滤掉多音字的有效变体
        if consider_polyphones:
            valid_polyphone_subs = []
            for op in substitutions:
                ref_char = ref_text[op[1]]
                hyp_char = hyp_text[op[2]]
                
                # 获取字符的拼音
                ref_pinyin = lazy_pinyin([ref_char], style=pypinyin.NORMAL)[0]
                hyp_pinyin = lazy_pinyin([hyp_char], style=pypinyin.NORMAL)[0]
                
                # 检查是否是多音字
                if ref_char in self.polyphone_dict:
                    if hyp_pinyin in self.polyphone_dict[ref_char]['normalized']:
                        valid_polyphone_subs.append(op)
            
            # 从substitutions中移除有效的多音字变体
            for op in valid_polyphone_subs:
                if op in substitutions:
                    substitutions.remove(op)
        
        # 构建详细错误分析
        char_errors = {
            'substitutions': {
                'count': len(substitutions),
                'details': []
            },
            'insertions': {
                'count': len(insertions),
                'details': []
            },
            'deletions': {
                'count': len(deletions),
                'details': []
            },
            'total_operations': len(operations)
        }
        
        # 添加详细信息
        for op in substitutions:
            char_errors['substitutions']['details'].append({
                'position': op[1],
                'ref_char': ref_text[op[1]],
                'hyp_char': hyp_text[op[2]]
            })
            
        for op in insertions:
            char_errors['insertions']['details'].append({
                'position': op[1],
                'hyp_char': hyp_text[op[2]]
            })
            
        for op in deletions:
            char_errors['deletions']['details'].append({
                'position': op[1],
                'ref_char': ref_text[op[1]]
            })
            
        return char_errors
    
    def _analyze_pinyin_errors(self, ref_text, hyp_text, consider_polyphones=True):
        """
        分析拼音级别的错误，特别处理多音字情况
        
        Args:
            ref_text: 参考文本
            hyp_text: 转写文本
            consider_polyphones: 是否考虑多音字
            
        Returns:
            拼音错误分析
        """
        # 提取拼音
        ref_pinyin = lazy_pinyin(ref_text, style=pypinyin.NORMAL)
        hyp_pinyin = lazy_pinyin(hyp_text, style=pypinyin.NORMAL)
        
        # 带声调拼音用于分析声调错误
        ref_pinyin_with_tone = lazy_pinyin(ref_text, style=pypinyin.TONE)
        hyp_pinyin_with_tone = lazy_pinyin(hyp_text, style=pypinyin.TONE)
        
        # 考虑多音字的情况
        if consider_polyphones:
            # 对参考文本中的多音字，使用所有可能的拼音变体计算最小编辑距离
            min_distance = float('inf')
            best_ref_pinyin = ref_pinyin
            
            # 生成多音字所有可能的拼音组合
            pinyin_variants = self._generate_pinyin_variants(ref_text)
            
            for variant in pinyin_variants:
                distance = Levenshtein.distance(''.join(variant), ''.join(hyp_pinyin))
                if distance < min_distance:
                    min_distance = distance
                    best_ref_pinyin = variant
                    
            ref_pinyin = best_ref_pinyin
        
        # 计算拼音编辑距离
        pinyin_distance = Levenshtein.distance(''.join(ref_pinyin), ''.join(hyp_pinyin))
        ref_pinyin_length = len(''.join(ref_pinyin))
        pinyin_error_rate = pinyin_distance / ref_pinyin_length if ref_pinyin_length > 0 else 0.0
        
        # 分析声调错误
        tone_errors = []
        min_len = min(len(ref_text), len(hyp_text))
        
        for i in range(min_len):
            # 基本拼音相同但声调不同
            if i < len(ref_pinyin) and i < len(hyp_pinyin) and \
               ref_pinyin[i] == hyp_pinyin[i] and \
               i < len(ref_pinyin_with_tone) and i < len(hyp_pinyin_with_tone) and \
               ref_pinyin_with_tone[i] != hyp_pinyin_with_tone[i]:
                tone_errors.append({
                    "position": i,
                    "ref_char": ref_text[i],
                    "hyp_char": hyp_text[i],
                    "ref_pinyin": ref_pinyin_with_tone[i],
                    "hyp_pinyin": hyp_pinyin_with_tone[i]
                })
        
        return {
            'pinyin_error_rate': pinyin_error_rate,
            'tone_errors': {
                'count': len(tone_errors),
                'details': tone_errors
            }
        }
    
    def _generate_pinyin_variants(self, text):
        """
        生成多音字的所有可能拼音组合
        
        Args:
            text: 输入文本
            
        Returns:
            拼音变体列表
        """
        # 生成单字拼音变体列表
        char_pinyin_variants = []
        for char in text:
            if char in self.polyphone_dict:
                # 使用多音字字典中的所有拼音变体
                char_pinyin_variants.append(self.polyphone_dict[char]['normalized'])
            else:
                # 非多音字使用标准拼音
                pinyin = lazy_pinyin([char], style=pypinyin.NORMAL)
                char_pinyin_variants.append(pinyin)
        
        # 限制变体数量，避免组合爆炸
        max_variants = 100
        total_combinations = 1
        for variants in char_pinyin_variants:
            total_combinations *= len(variants)
        
        if total_combinations <= max_variants:
            # 递归生成所有拼音组合
            return self._generate_combinations(char_pinyin_variants)
        else:
            # 超过限制，只使用默认拼音
            return [lazy_pinyin(text, style=pypinyin.NORMAL)]
    
    def _generate_combinations(self, variants_list, index=0, current=None):
        """
        递归生成拼音组合
        
        Args:
            variants_list: 每个位置的拼音变体列表
            index: 当前处理的索引
            current: 当前构建的组合
            
        Returns:
            所有可能的拼音组合
        """
        if current is None:
            current = []
            
        if index == len(variants_list):
            return [current.copy()]
            
        results = []
        for variant in variants_list[index]:
            current.append(variant)
            results.extend(self._generate_combinations(variants_list, index + 1, current))
            current.pop()
            
        return results
    
    def _analyze_word_errors(self, reference_text, transcribed_text):
        """
        分析词级别的错误
        
        Args:
            reference_text: 参考文本（原始未处理）
            transcribed_text: 转写文本（原始未处理）
            
        Returns:
            词错误分析
        """
        # 使用jieba分词
        ref_words = list(jieba.cut(reference_text))
        hyp_words = list(jieba.cut(transcribed_text))
        
        # 计算词错误率
        word_distance = Levenshtein.distance(ref_words, hyp_words)
        wer = word_distance / len(ref_words) if len(ref_words) > 0 else 0.0
        
        # 找出错误的词
        operations = Levenshtein.editops(ref_words, hyp_words)
        
        substitutions = []
        insertions = []
        deletions = []
        
        for op in operations:
            if op[0] == 'replace':
                substitutions.append({
                    'position': op[1],
                    'ref_word': ref_words[op[1]],
                    'hyp_word': hyp_words[op[2]]
                })
            elif op[0] == 'insert':
                insertions.append({
                    'position': op[1],
                    'hyp_word': hyp_words[op[2]]
                })
            elif op[0] == 'delete':
                deletions.append({
                    'position': op[1],
                    'ref_word': ref_words[op[1]]
                })
        
        return {
            'wer': wer,
            'ref_words': ref_words,
            'hyp_words': hyp_words,
            'substitutions': {
                'count': len(substitutions),
                'details': substitutions
            },
            'insertions': {
                'count': len(insertions),
                'details': insertions
            },
            'deletions': {
                'count': len(deletions),
                'details': deletions
            }
        }
    
    def _analyze_error_patterns(self, ref_text, hyp_text):
        """
        分析错误模式
        
        Args:
            ref_text: 参考文本
            hyp_text: 转写文本
            
        Returns:
            错误模式分析
        """
        # 获取编辑操作
        operations = Levenshtein.editops(ref_text, hyp_text)
        
        # 分析相似发音错误
        similar_sound_errors = []
        for op in operations:
            if op[0] == 'replace':
                ref_char = ref_text[op[1]]
                hyp_char = hyp_text[op[2]]
                
                # 获取拼音
                ref_pinyin = lazy_pinyin([ref_char], style=pypinyin.NORMAL)[0] if ref_char else ""
                hyp_pinyin = lazy_pinyin([hyp_char], style=pypinyin.NORMAL)[0] if hyp_char else ""
                
                # 判断拼音是否相似
                if self._is_similar_pinyin(ref_pinyin, hyp_pinyin):
                    similar_sound_errors.append({
                        'position': op[1],
                        'ref_char': ref_char,
                        'hyp_char': hyp_char,
                        'ref_pinyin': ref_pinyin,
                        'hyp_pinyin': hyp_pinyin
                    })
        
        # 分析多音字错误
        polyphone_errors = []
        for op in operations:
            if op[0] == 'replace':
                ref_char = ref_text[op[1]]
                hyp_char = hyp_text[op[2]]
                
                if ref_char in self.polyphone_dict:
                    ref_pinyin = lazy_pinyin([ref_char], style=pypinyin.NORMAL)[0]
                    hyp_pinyin = lazy_pinyin([hyp_char], style=pypinyin.NORMAL)[0]
                    
                    polyphone_errors.append({
                        'position': op[1],
                        'ref_char': ref_char,
                        'hyp_char': hyp_char,
                        'ref_pinyin': ref_pinyin,
                        'hyp_pinyin': hyp_pinyin,
                        'possible_pinyins': self.polyphone_dict[ref_char]['original']
                    })
        
        # 分析连续错误
        consecutive_errors = self._find_consecutive_errors(ref_text, hyp_text)
        
        return {
            'similar_sound_errors': {
                'count': len(similar_sound_errors),
                'details': similar_sound_errors
            },
            'polyphone_errors': {
                'count': len(polyphone_errors),
                'details': polyphone_errors
            },
            'consecutive_errors': {
                'count': len(consecutive_errors),
                'details': consecutive_errors
            }
        }
    
    def _is_similar_pinyin(self, pinyin1, pinyin2):
        """
        判断两个拼音是否相似
        
        Args:
            pinyin1: 第一个拼音
            pinyin2: 第二个拼音
            
        Returns:
            是否相似
        """
        if not pinyin1 or not pinyin2:
            return False
            
        # 计算拼音编辑距离
        distance = Levenshtein.distance(pinyin1, pinyin2)
        
        # 相似度阈值，可以根据需要调整
        threshold = 2
        
        return distance <= threshold
    
    def _find_consecutive_errors(self, ref_text, hyp_text):
        """
        查找连续错误
        
        Args:
            ref_text: 参考文本
            hyp_text: 转写文本
            
        Returns:
            连续错误列表
        """
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
        
    def analyze_batch_by_text_features(self, audios, sample_rates, texts):
        """
        按文本特征分析CER表现
        
        Args:
            audios: 音频数据列表
            sample_rates: 采样率列表
            texts: 文本列表
            
        Returns:
            按文本特征分组的CER分析
        """
        results = []
        
        for i, (audio, sr, text) in enumerate(zip(audios, sample_rates, texts)):
            cer = self.compute_cer(audio, sr, text)
            
            # 分析文本特征
            text_length = len(text)
            has_question = '?' in text
            has_polyphones = any(char in self.polyphone_dict for char in text)
            
            results.append({
                'text_id': i,
                'text': text,
                'cer': cer,
                'text_length': text_length,
                'has_question': has_question,
                'has_polyphones': has_polyphones
            })
        
        # 按文本长度分组
        length_groups = {
            'short (<=10)': [],
            'medium (11-30)': [],
            'long (>30)': []
        }
        
        for result in results:
            if result['text_length'] <= 10:
                length_groups['short (<=10)'].append(result['cer'])
            elif result['text_length'] <= 30:
                length_groups['medium (11-30)'].append(result['cer'])
            else:
                length_groups['long (>30)'].append(result['cer'])
        
        # 按是否包含问句分组
        question_groups = {
            'with_question': [],
            'without_question': []
        }
        
        for result in results:
            if result['has_question']:
                question_groups['with_question'].append(result['cer'])
            else:
                question_groups['without_question'].append(result['cer'])
        
        # 按是否包含多音字分组
        polyphone_groups = {
            'with_polyphones': [],
            'without_polyphones': []
        }
        
        for result in results:
            if result['has_polyphones']:
                polyphone_groups['with_polyphones'].append(result['cer'])
            else:
                polyphone_groups['without_polyphones'].append(result['cer'])
        
        # 计算各组平均CER
        analysis = {
            'by_length': {
                group: sum(cers)/len(cers) if cers else 0
                for group, cers in length_groups.items()
            },
            'by_question': {
                group: sum(cers)/len(cers) if cers else 0
                for group, cers in question_groups.items()
            },
            'by_polyphones': {
                group: sum(cers)/len(cers) if cers else 0
                for group, cers in polyphone_groups.items()
            },
            'overall': sum(result['cer'] for result in results) / len(results) if results else 0
        }
        
        return analysis


if __name__ == "__main__":
    import soundfile as sf
    import os
    
    # 设置音频文件路径
    #audio_dir = "tts_samples/"  # 存放TTS生成音频的目录
    
    # 准备测试文本和对应音频文件
    test_cases = [
        {
            "text": "今天天气真好，我很开心。",
            "audio_path": "听书-苏轼.wav"
        },
        #{
        #    "text": "长城是中国最著名的古迹，它的长度很长。",
        #    "audio_path": os.path.join(audio_dir, "sample2.wav")
        #},
        #{
        #    "text": "我买了一件衣服，花了几百元钱。",  # 包含多音字"几"和"长"
        #    "audio_path": os.path.join(audio_dir, "sample3.wav")
        #}
    ]
    
    # 准备音频列表和文本列表
    audios = []
    sample_rates = []
    texts = []
    
    # 加载所有音频文件
    for case in test_cases:
        try:
            # 使用soundfile加载音频
            audio_data, sample_rate = sf.read(case["audio_path"])
            
            # 如果是立体声，转为单声道
            if len(audio_data.shape) > 1 and audio_data.shape[1] > 1:
                audio_data = audio_data.mean(axis=1)
                
            audios.append(audio_data)
            sample_rates.append(sample_rate)
            texts.append(case["text"])
            
            print(f"成功加载音频: {case['audio_path']}, 采样率: {sample_rate}Hz, 时长: {len(audio_data)/sample_rate:.2f}秒")
        except Exception as e:
            print(f"加载音频 {case['audio_path']} 失败: {e}")
    
    # 初始化多音字感知CER评估器
    cer_calculator = FunASRCERCalculator(polyphone_path="polyphone.json")
    
    # 单个音频评估示例
    if audios:
        # 基本CER评估（第一个音频）
        print("\n=== 单个音频CER评估 ===")
        cer = cer_calculator.compute_cer(audios[0], sample_rates[0], texts[0])
        print(f"基本CER: {cer:.4f}")
        
        # 详细分析（第二个音频，如果存在）
        if len(audios) > 1:
            print("\n=== 详细CER分析（带多音字处理）===")
            detailed_result = cer_calculator.compute_cer(
                audios[1], sample_rates[1], texts[1], 
                consider_polyphones=True,
                detailed_analysis=True
            )
            
            # 输出转写结果
            print(f"原始文本: {texts[1]}")
            print(f"ASR转写: {detailed_result.get('transcription', '未获取到转写结果')}")
            print(f"CER: {detailed_result.get('cer', 0):.4f}")
            
            # 输出多音字分析
            print("\n多音字错误分析:")
            if 'error_patterns' in detailed_result and 'polyphone_errors' in detailed_result['error_patterns']:
                polyphone_errors = detailed_result['error_patterns']['polyphone_errors']
                if 'details' in polyphone_errors and polyphone_errors['details']:
                    for error in polyphone_errors['details']:
                        print(f"位置 {error['position']}: 参考字符 '{error['ref_char']}' " 
                              f"(拼音: {error['ref_pinyin']}) -> "
                              f"识别字符 '{error['hyp_char']}' (拼音: {error['hyp_pinyin']})")
                        print(f"  可能的拼音变体: {', '.join(error.get('possible_pinyins', []))}")
                else:
                    print("  未检测到多音字错误")
            else:
                print("  未包含多音字错误分析")
        
        # 批量特征分析
        if len(audios) > 2:
            print("\n=== 批量特征分析 ===")
            feature_analysis = cer_calculator.analyze_batch_by_text_features(
                audios, sample_rates, texts
            )
            
            print("\n文本特征CER分析:")
            print(f"整体CER: {feature_analysis['overall']:.4f}")
            print("按文本长度:")
            for group, cer in feature_analysis['by_length'].items():
                print(f"  {group}: {cer:.4f}")
            print("按是否包含多音字:")
            for group, cer in feature_analysis['by_polyphones'].items():
                print(f"  {group}: {cer:.4f}")
            print("按是否包含问句:")
            for group, cer in feature_analysis['by_question'].items():
                print(f"  {group}: {cer:.4f}")
    else:
        print("没有成功加载任何音频文件，请检查文件路径和格式")
