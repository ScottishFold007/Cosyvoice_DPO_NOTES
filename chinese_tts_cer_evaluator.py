import os
import numpy as np
import tempfile
import soundfile as sf
import json
from funasr import AutoModel  # 导入FunASR的AutoModel
import pypinyin
from pypinyin import lazy_pinyin, Style
import re
import jieba
import Levenshtein

class FunASRCERCalculator:
    """基于FunASR的中文TTS字符错误率评估器，支持多音字处理"""
    
    def __init__(self, model_name="paraformer-zh", 
                 polyphone_path="polyphone.json", 
                 device="cuda:0"):
        """
        初始化CER计算器
        Args:
            model_name: FunASR模型名称
            polyphone_path: 多音字词典路径
            device: 推理设备，默认GPU
        """
        # 初始化FunASR的ASR模型
        self.asr_model = AutoModel(model=model_name, device=device)
        
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
                        'original': pinyins,
                        'normalized': normalized_pinyins
                    }
            
            print(f"成功加载多音字字典，包含 {len(polyphone_dict)} 个条目")
            return polyphone_dict
        except Exception as e:
            print(f"加载多音字字典失败: {e}")
            return {}
    
    def _preprocess_text(self, text, ignore_punctuation=False, ignore_space=False, lowercase=True):
        """
        预处理文本，可选择性忽略标点、空格和大小写
        
        Args:
            text: 输入文本
            ignore_punctuation: 是否忽略标点符号
            ignore_space: 是否忽略空格
            lowercase: 是否转为小写
        
        Returns:
            处理后的文本
        """
        # 转小写处理
        if lowercase:
            text = text.lower()
            
        if ignore_punctuation:
            # 定义中英文标点符号
            punctuation = r"""，。！？；：""''（）【】《》、…—～·"""
            english_punctuation = r""",.!?;:"'()[]<>/\|`~@#$%^&*_+-="""
            # 移除所有标点符号
            for p in punctuation + english_punctuation:
                text = text.replace(p, '')
        
        if ignore_space:
            # 移除所有空格
            text = text.replace(' ', '')
        else:
            # 标准化空格：将连续多个空格替换为单个空格
            text = re.sub(r'\s+', ' ', text).strip()
        
        return text

    def compute_cer(self, audio, sample_rate, reference_text, 
                    consider_polyphones=False, detailed_analysis=False,
                    ignore_punctuation=True, ignore_space=True, lowercase=True):
        """
        计算字符错误率(CER)
        
        Args:
            audio: 音频数据
            sample_rate: 采样率
            reference_text: 参考文本
            consider_polyphones: 是否考虑多音字
            detailed_analysis: 是否进行详细分析
            ignore_punctuation: 是否忽略标点符号
            ignore_space: 是否忽略空格
            lowercase: 是否忽略大小写
        
        Returns:
            CER值或详细分析结果
        """
        # 获取ASR转写结果
        transcription = self._transcribe_audio(audio, sample_rate)
        
        # 预处理参考文本和转写文本
        processed_reference = self._preprocess_text(reference_text, 
                                                   ignore_punctuation=ignore_punctuation, 
                                                   ignore_space=ignore_space,
                                                   lowercase=lowercase)
        processed_transcription = self._preprocess_text(transcription, 
                                                       ignore_punctuation=ignore_punctuation, 
                                                       ignore_space=ignore_space,
                                                       lowercase=lowercase)
        
        # 计算标准CER
        standard_cer = self._calculate_cer(processed_reference, processed_transcription)
        
        # 计算考虑多音字的CER
        polyphone_cer = standard_cer
        operations = []
        if consider_polyphones:
            polyphone_cer_result = self._calculate_polyphone_aware_cer(
                processed_reference, processed_transcription)
            if isinstance(polyphone_cer_result, tuple):
                polyphone_cer, operations = polyphone_cer_result
            else:
                polyphone_cer = polyphone_cer_result
        
        # 计算词错误率WER
        wer = self._calculate_wer(processed_reference, processed_transcription)
        
        results = {
            'cer': polyphone_cer if consider_polyphones else standard_cer,
            'standard_cer': standard_cer,
            'wer': wer,
            'transcription': transcription,
            'processed_reference': processed_reference,
            'processed_transcription': processed_transcription,
        }
        
        # 如果需要详细分析，添加更多信息
        if detailed_analysis:
            # 添加字符级别错误分析
            char_level_analysis = self._analyze_char_level_errors(
                processed_reference, processed_transcription, consider_polyphones)
            results['char_level'] = char_level_analysis
            
            # 添加词级别错误分析
            word_level_analysis = self._analyze_word_level_errors(
                processed_reference, processed_transcription)
            results['word_level'] = word_level_analysis
            
            # 添加错误模式分析
            error_patterns = self._analyze_error_patterns(
                processed_reference, processed_transcription, consider_polyphones)
            results['error_patterns'] = error_patterns
        
        return results

    def _calculate_wer(self, reference, hypothesis):
        """
        计算词错误率(WER)
        
        Args:
            reference: 参考文本
            hypothesis: 识别文本
        
        Returns:
            WER值
        """
        # 分词处理
        import jieba
        
        # 对中英混合文本进行特殊处理
        def mixed_segment(text):
            # 预处理英文单词，防止被jieba分割
            # 查找所有英文单词
            import re
            english_words = re.findall(r'[a-zA-Z]+', text)
            # 为每个英文单词添加特殊标记
            marked_text = text
            for word in english_words:
                # 确保只替换完整的单词而不是单词的一部分
                marked_text = re.sub(r'\b' + word + r'\b', f"EN_WORD_{word}_EN_WORD", marked_text)
            
            # 使用jieba分词
            segments = jieba.cut(marked_text)
            result = []
            for seg in segments:
                # 恢复英文单词
                if seg.startswith("EN_WORD_") and seg.endswith("_EN_WORD"):
                    word = seg[8:-9]  # 去除标记
                    result.append(word)
                else:
                    result.append(seg)
            return result
        
        # 分词
        ref_words = mixed_segment(reference)
        hyp_words = mixed_segment(hypothesis)
        
        # 计算编辑距离
        distance = Levenshtein.distance(' '.join(ref_words), ' '.join(hyp_words))
        
        # 计算WER
        if len(ref_words) == 0:
            return 1.0 if len(hyp_words) > 0 else 0.0
        return distance / len(ref_words)

    def _analyze_word_level_errors(self, reference, hypothesis):
        """
        分析词级别错误
        
        Args:
            reference: 参考文本
            hypothesis: 识别文本
        
        Returns:
            词级别错误分析
        """
        import jieba
        
        # 分词
        ref_words = list(jieba.cut(reference))
        hyp_words = list(jieba.cut(hypothesis))
        
        # 计算编辑操作
        operations = Levenshtein.opcodes(' '.join(ref_words), ' '.join(hyp_words))
        
        # 统计词级别错误
        substitutions = 0
        deletions = 0
        insertions = 0
        
        for tag, i1, i2, j1, j2 in operations:
            if tag == 'replace':
                substitutions += (i2 - i1)
            elif tag == 'delete':
                deletions += (i2 - i1)
            elif tag == 'insert':
                insertions += (j2 - j1)
        
        total_words = len(ref_words)
        word_error_rate = (substitutions + deletions + insertions) / total_words if total_words > 0 else 0
        
        return {
            'substitutions': substitutions,
            'deletions': deletions,
            'insertions': insertions,
            'total_words': total_words,
            'word_error_rate': word_error_rate
        }


    def _transcribe_audio(self, audio, sample_rate):
        """
        使用FunASR模型转写音频
        
        Args:
            audio: 音频数据（numpy数组）
            sample_rate: 采样率
            
        Returns:
            转写文本
        """
        try:
            # 保存音频到临时文件
            temp_audio_path = os.path.join(self.temp_dir, "temp_audio.wav")
            sf.write(temp_audio_path, audio, sample_rate)
            
            # 使用FunASR模型进行转写
            result = self.asr_model.generate(input=temp_audio_path)
            
            # 获取转写文本
            transcription = result[0]["text"]
            
            # 删除临时文件
            if os.path.exists(temp_audio_path):
                os.remove(temp_audio_path)
                
            return transcription
        except Exception as e:
            print(f"音频转写失败: {e}")
            return ""


    def _calculate_cer(self, reference, hypothesis):
        """
        计算标准字符错误率
        
        Args:
            reference: 参考文本
            hypothesis: 识别文本
            
        Returns:
            CER值
        """
        # 使用Levenshtein距离计算编辑距离
        edit_distance = Levenshtein.distance(reference, hypothesis)
        
        # 计算CER
        if len(reference) == 0:
            return 1.0 if len(hypothesis) > 0 else 0.0
        
        return edit_distance / len(reference)

      
    def _calculate_polyphone_aware_cer(self, ref_text, hyp_text):
        """
        考虑多音字的CER计算
        
        Args:
            ref_text: 参考文本
            hyp_text: 假设文本
            
        Returns:
            cer值和编辑操作列表
        """
        # 获取参考文本和假设文本的拼音序列
        ref_pinyin = lazy_pinyin(ref_text)
        hyp_pinyin = lazy_pinyin(hyp_text)
        
        # 保存编辑操作
        operations = []
        
        # 初始化编辑距离矩阵
        m, n = len(ref_text), len(hyp_text)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        # 初始化第一行和第一列
        for i in range(m + 1):
            dp[i][0] = i
        for j in range(n + 1):
            dp[0][j] = j
        
        # 填充编辑距离矩阵
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                # 检查字符是否相同
                ref_char = ref_text[i-1]
                hyp_char = hyp_text[j-1]
                
                # 如果字符相同，不需要编辑
                if ref_char == hyp_char:
                    dp[i][j] = dp[i-1][j-1]
                else:
                    # 检查多音字情况：字不同但发音相同
                    ref_char_pinyin = ref_pinyin[i-1] if i-1 < len(ref_pinyin) else ''
                    hyp_char_pinyin = hyp_pinyin[j-1] if j-1 < len(hyp_pinyin) else ''
                    
                    # 如果拼音相同，且是正确的多音字变体
                    if (ref_char_pinyin == hyp_char_pinyin and 
                        ref_char_pinyin and 
                        self._is_valid_polyphone_variant(ref_char, hyp_char)):
                        # 这种情况下，不计算错误
                        dp[i][j] = dp[i-1][j-1]
                    else:
                        # 否则，取增删改三种操作的最小代价
                        dp[i][j] = min(
                            dp[i-1][j] + 1,     # 删除
                            dp[i][j-1] + 1,     # 插入
                            dp[i-1][j-1] + 1    # 替换
                        )
        
        # 反向回溯确定编辑操作
        i, j = m, n
        while i > 0 or j > 0:
            if i > 0 and j > 0 and ref_text[i-1] == hyp_text[j-1]:
                # 匹配
                operations.append(('match', i-1, j-1))
                i -= 1
                j -= 1
            elif i > 0 and j > 0 and dp[i][j] == dp[i-1][j-1] + 1:
                # 替换
                if i-1 < len(ref_pinyin) and j-1 < len(hyp_pinyin):
                    if ref_pinyin[i-1] == hyp_pinyin[j-1]:
                        operations.append(('polyphone', i-1, j-1))
                    else:
                        operations.append(('substitute', i-1, j-1))
                else:
                    operations.append(('substitute', i-1, j-1))
                i -= 1
                j -= 1
            elif i > 0 and dp[i][j] == dp[i-1][j] + 1:
                # 删除
                operations.append(('delete', i-1, -1))
                i -= 1
            elif j > 0 and dp[i][j] == dp[i][j-1] + 1:
                # 插入
                operations.append(('insert', -1, j-1))
                j -= 1
            else:
                # 多音字匹配情况
                operations.append(('polyphone', i-1, j-1))
                i -= 1
                j -= 1
        
        # 计算CER
        cer = dp[m][n] / m if m > 0 else 1.0
        
        return cer, operations
    
    def _is_valid_polyphone_variant(self, ref_char, hyp_char):
        """
        检查两个字符是否互为合法的多音字变体
        
        Args:
            ref_char: 参考字符
            hyp_char: 假设字符
            
        Returns:
            是否为合法多音字变体
        """
        # 检查参考字符是否是多音字
        if ref_char not in self.polyphone_dict:
            return False
        
        # 获取假设字符的拼音
        hyp_char_pinyin = lazy_pinyin(hyp_char)[0] if hyp_char else ''
        
        # 标准化拼音（移除声调）
        normalized_hyp_pinyin = re.sub(r'[àáâãäåāǎ]', 'a', hyp_char_pinyin)
        normalized_hyp_pinyin = re.sub(r'[èéêëēěẽ]', 'e', normalized_hyp_pinyin)
        normalized_hyp_pinyin = re.sub(r'[ìíîïīǐĩ]', 'i', normalized_hyp_pinyin)
        normalized_hyp_pinyin = re.sub(r'[òóôõöōǒ]', 'o', normalized_hyp_pinyin)
        normalized_hyp_pinyin = re.sub(r'[ùúûüūǔũ]', 'u', normalized_hyp_pinyin)
        normalized_hyp_pinyin = re.sub(r'[ǚǜǘǖü]', 'v', normalized_hyp_pinyin)
        normalized_hyp_pinyin = normalized_hyp_pinyin.lower()
        
        # 检查假设字符的拼音是否在参考字符的多音字列表中
        return normalized_hyp_pinyin in self.polyphone_dict[ref_char]['normalized']
    
    def _contains_polyphones(self, text):
        """
        检查文本是否包含多音字
        
        Args:
            text: 待检查的文本
            
        Returns:
            bool: 是否包含多音字
        """
        for char in text:
            if char in self.polyphone_dict:
                return True
        return False
    
    def _analyze_errors(self, ref_text, hyp_text, operations):
        """
        分析错误类型和模式
        
        Args:
            ref_text: 参考文本
            hyp_text: 假设文本
            operations: 编辑操作列表
            
        Returns:
            错误分析结果
        """
        # 初始化各类错误计数
        error_counts = {
            'substitutions': 0,
            'insertions': 0,
            'deletions': 0,
            'polyphone_errors': 0
        }
        
        # 详细错误分析
        char_level_errors = []
        polyphone_errors = {'count': 0, 'details': []}
        similar_sound_errors = {'count': 0, 'details': []}
        
        if operations:
            for op, ref_idx, hyp_idx in operations:
                if op == 'substitute':
                    error_counts['substitutions'] += 1
                    if ref_idx >= 0 and ref_idx < len(ref_text) and hyp_idx >= 0 and hyp_idx < len(hyp_text):
                        ref_char = ref_text[ref_idx]
                        hyp_char = hyp_text[hyp_idx]
                        
                        char_level_errors.append({
                            'type': 'substitute',
                            'position': ref_idx,
                            'ref_char': ref_char,
                            'hyp_char': hyp_char
                        })
                        
                        # 检查是否是发音相似错误
                        ref_char_pinyin = lazy_pinyin(ref_char)[0] if ref_char else ''
                        hyp_char_pinyin = lazy_pinyin(hyp_char)[0] if hyp_char else ''
                        
                        # 标准化拼音用于比较
                        ref_pinyin_norm = self._normalize_pinyin(ref_char_pinyin)
                        hyp_pinyin_norm = self._normalize_pinyin(hyp_char_pinyin)
                        
                        # 如果声母相同或拼音相似度大于阈值
                        if ref_pinyin_norm and hyp_pinyin_norm:
                            if self._get_pinyin_initial(ref_pinyin_norm) == self._get_pinyin_initial(hyp_pinyin_norm):
                                similar_sound_errors['count'] += 1
                                similar_sound_errors['details'].append({
                                    'position': ref_idx,
                                    'ref_char': ref_char,
                                    'hyp_char': hyp_char,
                                    'ref_pinyin': ref_char_pinyin,
                                    'hyp_pinyin': hyp_char_pinyin
                                })
                        
                elif op == 'delete':
                    error_counts['deletions'] += 1
                    if ref_idx >= 0 and ref_idx < len(ref_text):
                        char_level_errors.append({
                            'type': 'delete',
                            'position': ref_idx,
                            'ref_char': ref_text[ref_idx],
                            'hyp_char': ''
                        })
                elif op == 'insert':
                    error_counts['insertions'] += 1
                    if hyp_idx >= 0 and hyp_idx < len(hyp_text):
                        char_level_errors.append({
                            'type': 'insert',
                            'position': ref_idx if ref_idx >= 0 else 0,
                            'ref_char': '',
                            'hyp_char': hyp_text[hyp_idx]
                        })
                elif op == 'polyphone':
                    error_counts['polyphone_errors'] += 1
                    if ref_idx >= 0 and ref_idx < len(ref_text) and hyp_idx >= 0 and hyp_idx < len(hyp_text):
                        ref_char = ref_text[ref_idx]
                        hyp_char = hyp_text[hyp_idx]
                        
                        # 获取多音字的所有拼音变体
                        possible_pinyins = []
                        if ref_char in self.polyphone_dict:
                            possible_pinyins = self.polyphone_dict[ref_char]['original']
                        
                        polyphone_errors['details'].append({
                            'position': ref_idx,
                            'ref_char': ref_char,
                            'hyp_char': hyp_char,
                            'ref_pinyin': lazy_pinyin(ref_char, style=pypinyin.STYLE_TONE)[0] if ref_char else '',
                            'hyp_pinyin': lazy_pinyin(hyp_char, style=pypinyin.STYLE_TONE)[0] if hyp_char else '',
                            'possible_pinyins': possible_pinyins
                        })
        
        polyphone_errors['count'] = error_counts['polyphone_errors']
        
        # 词级别错误分析
        ref_words = list(jieba.cut(ref_text))
        hyp_words = list(jieba.cut(hyp_text))
        
        word_level_match = Levenshtein.ratio(
            ' '.join(ref_words), 
            ' '.join(hyp_words)
        )
        
        # 检测连续错误
        consecutive_errors = self._find_consecutive_errors(char_level_errors)
        
        # 检测声调错误
        tone_errors = self._find_tone_errors(ref_text, hyp_text)
        
        # 返回完整的错误分析结果
        return {
            'char_level': char_level_errors,
            'error_counts': error_counts,
            'word_level_match': word_level_match,
            'polyphone_errors': polyphone_errors,
            'similar_sound_errors': similar_sound_errors,
            'consecutive_errors': consecutive_errors,
            'tone_errors': tone_errors
        }
    
    def _normalize_pinyin(self, pinyin):
        """标准化拼音（移除声调等）"""
        if not pinyin:
            return ''
        
        # 去除音调
        normalized = re.sub(r'[àáâãäåāǎ]', 'a', pinyin)
        normalized = re.sub(r'[èéêëēěẽ]', 'e', normalized)
        normalized = re.sub(r'[ìíîïīǐĩ]', 'i', normalized)
        normalized = re.sub(r'[òóôõöōǒ]', 'o', normalized)
        normalized = re.sub(r'[ùúûüūǔũ]', 'u', normalized)
        normalized = re.sub(r'[ǚǜǘǖü]', 'v', normalized)
        
        return normalized.lower()
    
    def _get_pinyin_initial(self, pinyin):
        """获取拼音的声母"""
        initials = ['b', 'p', 'm', 'f', 'd', 't', 'n', 'l', 'g', 'k', 'h', 
                   'j', 'q', 'x', 'zh', 'ch', 'sh', 'r', 'z', 'c', 's', 'y', 'w']
        
        for initial in sorted(initials, key=len, reverse=True):
            if pinyin.startswith(initial):
                return initial
        
        return pinyin[0] if pinyin else ''
    
    def _find_consecutive_errors(self, char_errors):
        """查找连续错误"""
        if not char_errors:
            return []
        
        # 按位置排序错误
        sorted_errors = sorted(char_errors, key=lambda x: x['position'])
        
        consecutive_errors = []
        current_group = [sorted_errors[0]]
        
        for i in range(1, len(sorted_errors)):
            current_error = sorted_errors[i]
            prev_error = sorted_errors[i-1]
            
            # 如果位置相邻，添加到当前组
            if current_error['position'] - prev_error['position'] == 1:
                current_group.append(current_error)
            else:
                # 如果不相邻且当前组长度大于1，保存当前组
                if len(current_group) > 1:
                    consecutive_errors.append(current_group)
                # 开始新组
                current_group = [current_error]
        
        # 添加最后一组（如果长度大于1）
        if len(current_group) > 1:
            consecutive_errors.append(current_group)
            
        return consecutive_errors
    
    def _find_tone_errors(self, ref_text, hyp_text):
        """查找声调错误"""
        # 提取带声调的拼音
        ref_pinyin_with_tone = lazy_pinyin(ref_text, style=pypinyin.STYLE_TONE)
        hyp_pinyin_with_tone = lazy_pinyin(hyp_text, style=pypinyin.STYLE_TONE)
        
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
    
    def _analyze_char_level_errors(self, ref_text, hyp_text, consider_polyphones=False):
        """
        分析字符级别的错误
        
        Args:
            ref_text: 参考文本
            hyp_text: 假设文本
            consider_polyphones: 是否考虑多音字
            
        Returns:
            字符级别错误分析结果
        """
        # 计算编辑距离和操作
        if consider_polyphones:
            _, operations = self._calculate_polyphone_aware_cer(ref_text, hyp_text)
        else:
            # 使用标准编辑距离
            operations = Levenshtein.editops(ref_text, hyp_text)
            # 转换为与多音字处理兼容的格式
            formatted_ops = []
            for op, ref_idx, hyp_idx in operations:
                if op == 'replace':
                    formatted_ops.append(('substitute', ref_idx, hyp_idx))
                else:
                    formatted_ops.append((op, ref_idx, hyp_idx))
            operations = formatted_ops
        
        # 分析错误
        return self._analyze_errors(ref_text, hyp_text, operations)

    def _analyze_error_patterns(self, ref_text, hyp_text, consider_polyphones=False):
        """
        分析错误模式
        
        Args:
            ref_text: 参考文本
            hyp_text: 假设文本
            consider_polyphones: 是否考虑多音字
            
        Returns:
            错误模式分析
        """
        # 获取字符级别错误分析
        char_analysis = self._analyze_char_level_errors(ref_text, hyp_text, consider_polyphones)
        
        # 提取错误计数
        error_counts = char_analysis.get('error_counts', {})
        
        # 计算错误率
        total_chars = len(ref_text)
        error_rates = {
            'substitution_rate': error_counts.get('substitutions', 0) / total_chars if total_chars > 0 else 0,
            'deletion_rate': error_counts.get('deletions', 0) / total_chars if total_chars > 0 else 0,
            'insertion_rate': error_counts.get('insertions', 0) / total_chars if total_chars > 0 else 0,
            'polyphone_error_rate': error_counts.get('polyphone_errors', 0) / total_chars if total_chars > 0 else 0
        }
        
        # 分析常见错误模式
        common_errors = {}
        char_level_errors = char_analysis.get('char_level', [])
        
        for error in char_level_errors:
            if error['type'] == 'substitute':
                error_pair = (error['ref_char'], error['hyp_char'])
                if error_pair in common_errors:
                    common_errors[error_pair] += 1
                else:
                    common_errors[error_pair] = 1
        
        # 按频率排序常见错误
        sorted_common_errors = sorted(
            common_errors.items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        
        # 提取前10个最常见错误
        top_errors = [
            {'ref': pair[0][0], 'hyp': pair[0][1], 'count': pair[1]} 
            for pair in sorted_common_errors[:10]
        ]
        
        # 返回错误模式分析
        return {
            'error_rates': error_rates,
            'top_common_errors': top_errors,
            'consecutive_errors': char_analysis.get('consecutive_errors', []),
            'similar_sound_errors': char_analysis.get('similar_sound_errors', {}),
            'tone_errors': char_analysis.get('tone_errors', [])
        }

    def compute_batch_cer(self, audios, sample_rates, texts, consider_polyphones=True,
                           ignore_punctuation=True, ignore_space=True, lowercase=True):
        """
        批量计算CER
        
        Args:
            audios: 音频数据列表
            sample_rates: 采样率列表
            texts: 文本列表
            consider_polyphones: 是否考虑多音字
            ignore_punctuation: 是否忽略标点符号
            ignore_space: 是否忽略空格
            lowercase: 是否忽略大小写
            
        Returns:
            CER值列表
        """
        results = []
        for audio, sr, text in zip(audios, sample_rates, texts):
            cer = self.compute_cer(
                audio, sr, text, 
                consider_polyphones=consider_polyphones,
                ignore_punctuation=ignore_punctuation,
                ignore_space=ignore_space,
                lowercase=lowercase
            )
            results.append(cer)
        
        return results
    
    def analyze_batch_results(self, audios, sample_rates, texts, consider_polyphones=True,
                               ignore_punctuation=True, ignore_space=True, lowercase=True):
        """
        批量计算CER并返回详细分析
        
        Args:
            audios: 音频数据列表
            sample_rates: 采样率列表
            texts: 文本列表
            consider_polyphones: 是否考虑多音字
            ignore_punctuation: 是否忽略标点符号
            ignore_space: 是否忽略空格
            lowercase: 是否忽略大小写
            
        Returns:
            详细分析结果列表
        """
        results = []
        for audio, sr, text in zip(audios, sample_rates, texts):
            result = self.compute_cer(
                audio, sr, text, 
                consider_polyphones=consider_polyphones,
                detailed_analysis=True,
                ignore_punctuation=ignore_punctuation,
                ignore_space=ignore_space,
                lowercase=lowercase
            )
            results.append(result)
        
        return results
    
    def analyze_batch_by_text_features(self, audios, sample_rates, texts, 
                                       consider_polyphones=True,
                                       ignore_punctuation=True, 
                                       ignore_space=True, 
                                       lowercase=True):
        """
        按文本特征分析CER
        
        Args:
            audios: 音频数据列表
            sample_rates: 采样率列表
            texts: 文本列表
            consider_polyphones: 是否考虑多音字
            ignore_punctuation: 是否忽略标点符号
            ignore_space: 是否忽略空格
            lowercase: 是否忽略大小写
            
        Returns:
            按文本特征分组的CER分析
        """
        # 获取详细结果
        results = self.analyze_batch_results(
            audios, sample_rates, texts, 
            consider_polyphones=consider_polyphones,
            ignore_punctuation=ignore_punctuation,
            ignore_space=ignore_space,
            lowercase=lowercase
        )
        
        # 按文本长度分组
        length_groups = {
            'short (<=10)': [],
            'medium (11-30)': [],
            'long (>30)': []
        }
        
        # 添加文本长度字段
        for i, result in enumerate(results):
            text_length = len(self._preprocess_text(texts[i], 
                                                 ignore_punctuation=ignore_punctuation,
                                                 ignore_space=ignore_space))
            result['text_length'] = text_length
            
            # 按长度分类
            if text_length <= 10:
                length_groups['short (<=10)'].append(result['cer'])
            elif text_length <= 30:
                length_groups['medium (11-30)'].append(result['cer'])
            else:
                length_groups['long (>30)'].append(result['cer'])
        
        # 按是否包含问句分组
        question_groups = {
            'with_question': [],
            'without_question': []
        }
        
        for result in results:
            # 检查原始文本是否包含问号
            has_question = '?' in texts[results.index(result)] or '？' in texts[results.index(result)]
            result['has_question'] = has_question
            
            if has_question:
                question_groups['with_question'].append(result['cer'])
            else:
                question_groups['without_question'].append(result['cer'])
        
        # 按是否包含多音字分组
        polyphone_groups = {
            'with_polyphones': [],
            'without_polyphones': []
        }
        
        for result in results:
            # 检查原始文本是否包含多音字
            has_polyphones = self._contains_polyphones(texts[results.index(result)])
            result['has_polyphones'] = has_polyphones
            
            if has_polyphones:
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

# 测试代码部分
if __name__ == "__main__":
    import soundfile as sf
    import os
    
    # 设置音频文件路径
    audio_dir = "tts_samples/"  # 存放TTS生成音频的目录
    
    # 准备测试文本和对应音频文件
    test_cases = [
        {
            "text": "Hi，你最近怎么样？我最近听说你在运作一个big project, 进展如何？记住，偶尔也要take a break，不要太push你自己了！ 毕竟，健康最重要，right？By the way, 周末有什么plan吗？要不要一起去那个新开的cafe坐坐，听说他们的latte和cake都超级delicious。",  # 修改为与音频内容匹配的文本
            "audio_path": "中英夹杂.wav"
        },
        # 可以添加更多测试案例以充分利用代码功能
    ]
    
    # 准备音频列表和文本列表
    audios = []
    sample_rates = []
    texts = []
    
    # 加载所有音频文件
    for case in test_cases:
        try:
            # 正确组合路径
            full_path = os.path.join(audio_dir, case["audio_path"])
            # 使用soundfile加载音频
            audio_data, sample_rate = sf.read(full_path)
            
            # 如果是立体声，转为单声道
            if len(audio_data.shape) > 1 and audio_data.shape[1] > 1:
                audio_data = audio_data.mean(axis=1)
            
            # 采样率检查（FunASR通常需要16kHz采样率）
            if sample_rate != 16000:
                print(f"警告：音频采样率为 {sample_rate}Hz，而FunASR最佳采样率为16000Hz")
                
            audios.append(audio_data)
            sample_rates.append(sample_rate)
            texts.append(case["text"])
            
            print(f"成功加载音频: {full_path}, 采样率: {sample_rate}Hz, 时长: {len(audio_data)/sample_rate:.2f}秒")
        except FileNotFoundError:
            print(f"错误：找不到音频文件 {full_path}，请确认文件路径")
        except Exception as e:
            print(f"加载音频 {full_path} 失败: {e}")
    
    # 初始化多音字感知CER评估器
    cer_calculator = FunASRCERCalculator(model_name="paraformer-zh", polyphone_path="polyphone.json")
    
    # 单个音频评估示例
    if audios:
        # 基本CER评估（使用优化后的参数）
        print("\n=== 单个音频CER评估 ===")
        cer_result = cer_calculator.compute_cer(
            audios[0], 
            sample_rates[0], 
            texts[0], 
            consider_polyphones=True,
            ignore_punctuation=True,
            ignore_space=True,
            lowercase=True
        )
        print(f"基本CER: {cer_result['cer']:.4f}")
        print(f"标准CER: {cer_result['standard_cer']:.4f}")
        print(f"WER: {cer_result['wer']:.4f}")
        
        # 输出处理后的文本，便于对比
        print(f"\n参考文本(处理后): {cer_result['processed_reference']}")
        print(f"转写文本(处理后): {cer_result['processed_transcription']}")
        print(f"\n原始转写文本: {cer_result['transcription']}")
        
        # 添加详细分析
        print("\n=== 详细CER分析 ===")
        detailed_result = cer_calculator.compute_cer(
            audios[0], 
            sample_rates[0], 
            texts[0], 
            consider_polyphones=True,
            detailed_analysis=True,
            ignore_punctuation=True,
            ignore_space=True,
            lowercase=True
        )
        
        # 输出详细错误分析
        if 'char_level' in detailed_result:
            char_level = detailed_result['char_level']
            error_counts = char_level.get('error_counts', {})
            
            print("\n字符级错误分析:")
            print(f"  替换错误: {error_counts.get('substitutions', 0)}")
            print(f"  删除错误: {error_counts.get('deletions', 0)}")
            print(f"  插入错误: {error_counts.get('insertions', 0)}")
            
            if 'error_patterns' in detailed_result:
                error_patterns = detailed_result['error_patterns']
                if 'top_common_errors' in error_patterns and error_patterns['top_common_errors']:
                    print("\n常见错误模式:")
                    for i, err in enumerate(error_patterns['top_common_errors'][:5]):
                        print(f"  {i+1}. '{err['ref']}' → '{err['hyp']}' (出现 {err['count']} 次)")
    else:
        print("没有成功加载任何音频文件，请检查文件路径和格式")


# 在输出详细错误分析部分后添加以下代码

# 1. 声调错误分析
if 'error_patterns' in detailed_result:
    error_patterns = detailed_result['error_patterns']
    
    # 声调错误分析
    if 'tone_errors' in error_patterns and error_patterns['tone_errors']:
        print("\n=== 声调错误分析 ===")
        print(f"发现 {len(error_patterns['tone_errors'])} 个声调错误")
        for i, err in enumerate(error_patterns['tone_errors'][:5]):  # 只显示前5个
            print(f"  {i+1}. 位置 {err['position']}: '{err['ref_char']}({err['ref_pinyin']})' → '{err['hyp_char']}({err['hyp_pinyin']})'")

# 2. 连续错误检测
if 'char_level' in detailed_result:
    char_level = detailed_result['char_level']
    
    # 连续错误检测
    if 'consecutive_errors' in char_level and char_level['consecutive_errors']:
        print("\n=== 连续错误检测 ===")
        print(f"发现 {len(char_level['consecutive_errors'])} 组连续错误")
        for i, group in enumerate(char_level['consecutive_errors'][:3]):  # 只显示前3组
            print(f"  连续错误组 {i+1} (长度: {len(group)}):")
            errors_text = []
            for err in group:
                if err['type'] == 'substitute':
                    errors_text.append(f"'{err['ref_char']}→{err['hyp_char']}'")
                elif err['type'] == 'delete':
                    errors_text.append(f"'{err['ref_char']}→∅'")
                elif err['type'] == 'insert':
                    errors_text.append(f"'∅→{err['hyp_char']}'")
            print(f"    错误序列: {' '.join(errors_text)}")

# 3. 发音相似错误分析
if 'error_patterns' in detailed_result:
    error_patterns = detailed_result['error_patterns']
    
    # 发音相似错误分析
    if 'similar_sound_errors' in error_patterns and error_patterns['similar_sound_errors'].get('count', 0) > 0:
        similar_errors = error_patterns['similar_sound_errors']
        print("\n=== 发音相似错误分析 ===")
        print(f"发现 {similar_errors['count']} 个发音相似错误")
        for i, err in enumerate(similar_errors.get('details', [])[:5]):  # 只显示前5个
            print(f"  {i+1}. '{err['ref_char']}({err['ref_pinyin']})' → '{err['hyp_char']}({err['hyp_pinyin']})'")

# 4. 添加更多分析功能的统计概览
print("\n=== 错误类型统计 ===")
if 'error_patterns' in detailed_result:
    error_rates = detailed_result['error_patterns'].get('error_rates', {})
    print(f"替换错误率: {error_rates.get('substitution_rate', 0):.2%}")
    print(f"删除错误率: {error_rates.get('deletion_rate', 0):.2%}")
    print(f"插入错误率: {error_rates.get('insertion_rate', 0):.2%}")
    
    # 计算各种特殊错误所占比例
    total_errors = sum(detailed_result['char_level'].get('error_counts', {}).values())
    if total_errors > 0:
        polyphone_count = detailed_result['char_level'].get('polyphone_errors', {}).get('count', 0)
        similar_count = detailed_result['error_patterns'].get('similar_sound_errors', {}).get('count', 0)
        tone_count = len(detailed_result['error_patterns'].get('tone_errors', []))
        
        print(f"多音字错误占比: {polyphone_count/total_errors:.2%}")
        print(f"发音相似错误占比: {similar_count/total_errors:.2%}")
        print(f"声调错误占比: {tone_count/total_errors:.2%}")
