import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# CSV 파일의 경로
file_path = 'data/train.csv'

# pandas를 사용하여 CSV 파일을 DataFrame으로 읽기
df = pd.read_csv(file_path)
# 'id' 열 삭제
df = df.drop(columns=['id'])

# DataFrame의 인덱스 열의 이름을 'id'로 변경
df.index.name = 'id'

# DataFrame 출력
sns.set(style="whitegrid")

# Label 별 분포를 시각화할 준비
plt.figure(figsize=(10, 6))
sns.histplot(df['label'], bins=20, kde=True)
plt.title('Distribution of Labels')
plt.xlabel('Label')
plt.ylabel('Frequency')
plt.show()