import pandas as pd

# 원본 데이터 로드
df = pd.read_csv('train.csv')

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

# 전체 데이터의 sentence_1과 sentence_2 순서를 바꾼 데이터 추가
swapped_df = df.copy()
swapped_df['sentence_1'], swapped_df['sentence_2'] = swapped_df['sentence_2'], swapped_df['sentence_1']

# swapped_df에 label을 기존과 동일하게 설정 (복사한 경우와는 다르게 레이블을 바꾸지 않음)
df_final = pd.concat([df_final, swapped_df], ignore_index=True)

# 최종 데이터셋 저장
df_final.to_csv('sec_aug_train.csv', index=False)
