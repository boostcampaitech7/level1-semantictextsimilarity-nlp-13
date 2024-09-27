import pandas as pd
import numpy as np
from scipy.stats import pearsonr
from itertools import product

# CSV 파일 읽기
output_1 = pd.read_csv('dev_output_for_ensemble/dev_output_kykim.csv')
output_2 = pd.read_csv('dev_output_for_ensemble/dev_output_lucid.csv')
output_3 = pd.read_csv('dev_output_for_ensemble/dev_output_snunlp.csv')
output_4 = pd.read_csv('dev_output_for_ensemble/dev_output_klue.csv')

# 실제 정답 불러오기 (output.csv의 label)
actual = output_1['label']

# 가중 앙상블 함수 정의
def weighted_ensemble(df_list, weights):
    ensemble_df = df_list[0].copy()
    weighted_sum = sum(df['target'] * w for df, w in zip(df_list, weights))
    ensemble_df['target'] = weighted_sum / sum(weights)
    return ensemble_df['target']

# Pearson 상관계수 계산
def calculate_pearson(predicted, actual):
    pearson_corr, _ = pearsonr(predicted, actual)
    return pearson_corr

# 가중치 리스트 정의 (가중치 후보: 0.1에서 1까지, 합이 1이 되는 조합만 사용)
weight_range = np.arange(0.01, 1.01, 0.01)
best_weights = None
best_pearson = -1



for w1, w2, w3 in product(weight_range, repeat=3):
    if abs(w1 + w2 + w3 - 1.0) < 1e-6:  # 가중치의 합이 1인 경우만 고려
        weights = [w1, w2, w3]
        
        # 가중 앙상블 적용
        predicted = weighted_ensemble([output_1, output_2, output_3], weights)
        
        # Pearson 상관계수 계산
        pearson_corr = calculate_pearson(predicted, actual)
        
        # Pearson 상관계수가 더 높으면 갱신
        if pearson_corr > best_pearson:
            best_pearson = pearson_corr
            best_weights = weights

'''
for w1, w2, w3, w4 in product(weight_range, repeat=4):
    if abs(w1 + w2 + w3 + w4 - 1.0) < 1e-6:  # 가중치의 합이 1인 경우만 고려
        weights = [w1, w2, w3, w4]
        
        # 가중 앙상블 적용
        predicted = weighted_ensemble([output_1, output_2, output_3, output_4], weights)
        
        # Pearson 상관계수 계산
        pearson_corr = calculate_pearson(predicted, actual)
        
        # Pearson 상관계수가 더 높으면 갱신
        if pearson_corr > best_pearson:
            best_pearson = pearson_corr
            best_weights = weights
'''
# 최종 결과 출력
print(f"Best Weights: {best_weights}")
print(f"Best Pearson Correlation: {best_pearson}")
