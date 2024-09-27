import pandas as pd

# 파일 불러오기
output_1 = pd.read_csv('test_output_for_ensemble/test_output_kykim.csv')
output_2 = pd.read_csv('test_output_for_ensemble/test_output_lucid.csv')
output_3 = pd.read_csv('test_output_for_ensemble/test_output_snunlp.csv')
output_4 = pd.read_csv('test_output_for_ensemble/test_output_klue.csv')
# 평균 앙상블
def average_ensemble(df_list):
    ensemble_df = df_list[0].copy()
    ensemble_df['target'] = sum(df['target'] for df in df_list) / len(df_list)
    return ensemble_df

# 가중 평균 앙상블 (각 모델에 가중치를 부여)
def weighted_ensemble(df_list, weights):
    ensemble_df = df_list[0].copy()
    weighted_sum = sum(df['target'] * w for df, w in zip(df_list, weights))
    ensemble_df['target'] = weighted_sum / sum(weights)
    return ensemble_df

# 앙상블 방식 선택: 평균 앙상블
# ensemble_result = average_ensemble([output_1, output_2, output_3])

# 또는 가중 앙상블 (가중치를 수정하여 조정 가능)
weights = [0.21,0.36,0.34,0.09]  # 예시: 각 모델의 가중치
ensemble_result = weighted_ensemble([output_1, output_2, output_3, output_4], weights)

# 결과를 output.csv로 저장
ensemble_result.to_csv('ensemble_output.csv', index=False)
