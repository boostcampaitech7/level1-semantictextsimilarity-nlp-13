import pandas as pd
import random
import re

# class로 만들자

class DataAugmenter:
    def __init__(self, replace_words=None):
        # 교체할 단어 목록을 초기화, 기본적으로 '너무', '진짜', '정말' 사용
        if replace_words is None:
            self.replace_words = ['너무', '진짜', '정말']
        else:
            self.replace_words = replace_words
    
    def augment(self, data, is_preprocessed=False):
        """
        주어진 데이터를 받아 문장 스왑과 특정 단어 교체로 데이터 증강을 수행한 후
        증강된 데이터를 반환하는 함수.
        
        Parameters:
            data (pd.DataFrame): 증강할 데이터 (sentence_1, sentence_2 열 포함)
            is_preprocessed (bool): 데이터가 띄어쓰기 교정된 데이터인지 여부
            
        Returns:
            pd.DataFrame: 증강된 데이터
        """
        # id와 source 등의 불필요한 열 제거 (데이터에 따라 필요 없을 경우 생략 가능)
        if 'id' in data.columns:
            data = data.drop(columns={'id'})
        if 'source' in data.columns:
            data = data.drop(columns={'source'})
        
        # 데이터 복사 후 s1과 s2를 스왑
        swapped_data = data.copy()
        swapped_data['sentence_1'], swapped_data['sentence_2'] = data['sentence_2'], data['sentence_1']
        
        # 원본 데이터와 스왑된 데이터를 결합
        combined_data = pd.concat([data, swapped_data], ignore_index=True)
        
        # '너무', '진짜', '정말' 중 하나가 포함된 문장을 필터링
        target_words = '|'.join(self.replace_words)
        filtered_data = combined_data[
            combined_data['sentence_1'].str.contains(target_words) | 
            combined_data['sentence_2'].str.contains(target_words)
        ].copy()
        
        # 단어 대체 함수 정의
        if is_preprocessed:
            # 띄어쓰기 교정된 데이터일 때
            def replace_word(sentence):
                words = sentence.split()
                for i, word in enumerate(words):
                    if word in self.replace_words:
                        # 현재 단어를 제외한 나머지 단어들 중 하나로 랜덤하게 선택하여 대체
                        choices = [w for w in self.replace_words if w != word]
                        words[i] = random.choice(choices)
                return ' '.join(words)
        else:
            # 띄어쓰기 교정되지 않은 데이터일 때
            def replace_word(sentence):
                pattern = r'(너무|진짜|정말)'
                def replace_match(match):
                    word = match.group(0)
                    choices = [w for w in self.replace_words if w != word]
                    return random.choice(choices)
                result = re.sub(pattern, replace_match, sentence)
                return result
        
        # sentence_1과 sentence_2에 대해 단어를 랜덤하게 대체
        filtered_data['sentence_1'] = filtered_data['sentence_1'].apply(replace_word)
        filtered_data['sentence_2'] = filtered_data['sentence_2'].apply(replace_word)
        
        # 원본 + 스왑 + 단어 대체 데이터를 결합하여 최종 증강 데이터 생성
        final_augmented_data = pd.concat([combined_data, filtered_data], ignore_index=True)
        
        return final_augmented_data


augmenter = DataAugmenter()

raw_train_data = pd.read_csv('data/train.csv')
df = augmenter.augment(raw_train_data, is_preprocessed=False)

# label이 0인 데이터 샘플링
df_0 = df[df['label'] == 0].sample(n=100, random_state=42)

# label이 0이 아닌 데이터
df_new = df[df['label'] != 0].copy()

# label이 0인 데이터를 df_new에 추가
df_new = pd.concat([df_new, df_0], ignore_index=True)

# label이 0인 데이터 복사 및 레이블 변경
copied_df = df[df['label'] == 0].sample(n=500, random_state=42).copy()
copied_df['sentence_1'] = copied_df['sentence_2']
copied_df['label'] = 5.0

# 최종 데이터셋에 추가
df_final = pd.concat([df_new, copied_df], ignore_index=True)

# label이 1점에서 3점인 데이터 선택 및 추가
augmented_data = df_final[(df_final['label'] >= 1) & (df_final['label'] <= 3)].copy()

# 기존 데이터에 복사된 데이터 추가
augmented_data = pd.concat([df_final, augmented_data], ignore_index=True)

# 결과 저장
augmented_data.to_csv('data/aug_train.csv', index=False)


