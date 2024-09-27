import pandas as pd

# # 파일 불러오기
output_1 = pd.read_csv('output/STSModel_kykim-electra-kor-base_val_pearson=0.9258827567100525/dev_output_kykim.csv')
output_2 = pd.read_csv('output/STSModel_klue-roberta-base_val_pearson=0.9215819835662842/dev_output_klue.csv')
output_3 = pd.read_csv('output/STSModel_eenzeenee-t5-base-korean-summarization_val_pearson=0.9229395985603333/dev_output_t5.csv')

# 가중 평균 앙상블 (각 모델에 가중치를 부여)
def weighted_ensemble(df_list, weights):
    ensemble_df = df_list[0].copy()
    weighted_sum = sum(df['target'] * w for df, w in zip(df_list, weights))
    ensemble_df['target'] = weighted_sum / sum(weights)
    return ensemble_df

# 또는 가중 앙상블 (가중치를 수정하여 조정 가능)
weights = [0.4, 0.27, 0.33]  # 예시: 각 모델의 가중치
dev_ensemble_result = weighted_ensemble([output_1, output_2, output_3], weights)

# 결과를 output.csv로 저장
dev_ensemble_result.to_csv('output/eng2_/dev_output_eng2.csv')

# # 파일 불러오기
output_1 = pd.read_csv('output/STSModel_kykim-electra-kor-base_val_pearson=0.9258827567100525/test_output_kykim.csv')
output_2 = pd.read_csv('output/STSModel_klue-roberta-base_val_pearson=0.9215819835662842/test_output_klue.csv')
output_3 = pd.read_csv('output/STSModel_eenzeenee-t5-base-korean-summarization_val_pearson=0.9229395985603333/test_output_t5.csv')

# 또는 가중 앙상블 (가중치를 수정하여 조정 가능)
weights = [0.4, 0.27, 0.33]  # 예시: 각 모델의 가중치
test_ensemble_result = weighted_ensemble([output_1, output_2, output_3], weights)

test_ensemble_result.to_csv('output/eng2_/test_output_eng2.csv', index=False)
