import pandas as pd
from scipy.stats import pearsonr

# 앙상블 결과 불러오기
ensemble_result = pd.read_csv('ensemble_output.csv')

# 예측값과 정답값 추출
predicted = ensemble_result['target']
actual = ensemble_result['label']

# Pearson 상관계수 계산
pearson_corr, p_value = pearsonr(predicted, actual)

print(f"Pearson Correlation Coefficient: {pearson_corr}")