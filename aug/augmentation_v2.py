import pandas as pd

# CSV 파일 읽기
df = pd.read_csv('aug_aug_train.csv')

# label이 1점에서 3점인 데이터 선택
augmented_data = df[(df['label'] >= 1) & (df['label'] <= 3)].copy()

# 데이터 복사하여 붙여넣기
augmented_data = pd.concat([df, augmented_data], ignore_index=True)

# 결과 저장
augmented_data.to_csv('aug_aug_aug_.csv', index=False)
