import pandas as pd
from sklearn.linear_model import Ridge  # 릿지 메타 모델 사용
from scipy.stats import pearsonr
from sklearn.model_selection import KFold, GridSearchCV
import numpy as np
from sklearn.preprocessing import StandardScaler


# 각 모델의 예측값 불러오기
kykim_electra_df = pd.read_csv('output/STSModel_kykim-electra-kor-base_val_pearson=0.9258827567100525/dev_output_kykim.csv')
deberta_df = pd.read_csv('output/STSModel_team-lucid-deberta-v3-base-korean_val_pearson=0.9326505064964294/dev_output_lucid.csv')
snu_nlp_df = pd.read_csv('output/STSModel_snunlp-KR-ELECTRA-discriminator_val_pearson=0.9332568049430847/dev_output_snunlp.csv')
eng2_df = pd.read_csv('output\eng2_\dev_output_eng2.csv')

# target 열의 값에서 숫자만 추출하는 함수
def extract_number(text):
    clean_text = str(text).replace('[', '').replace(']', '').strip()
    return float(clean_text)

# 필요한 경우 target 열의 값에서 숫자만 추출하여 새로운 열 생성
# deberta_df['target'] = deberta_df['target'].apply(extract_number)
# snu_nlp_df['target'] = snu_nlp_df['target'].apply(extract_number)

# 예측값을 하나의 데이터프레임으로 병합
stacked_df = pd.DataFrame({
    'kykim_preds': kykim_electra_df['target'],
    'deberta_preds': deberta_df['target'],
    'snunlp_preds': snu_nlp_df['target'],
    'eng2_preds': eng2_df['target'],
    'label': kykim_electra_df['label']  # 실제 label 값은 동일하다고 가정
})

# 메타 모델의 입력과 타겟 변수 설정
X = stacked_df[['kykim_preds', 'deberta_preds', 'snunlp_preds', 'eng2_preds']]  # 예측값을 입력으로 사용
y = stacked_df['label']

# 예측값 스케일링
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 교차 검증 설정: 폴드 수 증가 및 데이터 섞기
kf = KFold(n_splits=5, shuffle=True)
meta_predictions = np.zeros(len(X))  # 메타 모델의 예측값을 저장할 배열
pearson_scores = []

# 교차 검증을 통해 메타 모델 훈련 및 평가
param_grid = {
    'alpha': [0.01, 0.1, 1.0, 10.0, 100.0]  # Ridge 모델의 하이퍼파라미터 alpha 값 조정
}

for train_index, test_index in kf.split(X_scaled):
    X_train, X_test = X_scaled[train_index], X_scaled[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    
    # 메타 모델로 Ridge 사용
    grid_search = GridSearchCV(Ridge(), param_grid, cv=5)
    grid_search.fit(X_train, y_train)
    best_meta_model = grid_search.best_estimator_
    
    y_pred = best_meta_model.predict(X_test)
    meta_predictions[test_index] = y_pred  # 테스트 데이터에 대한 예측값을 저장
    
    pearson_corr, _ = pearsonr(y_pred, y_test)
    pearson_scores.append(pearson_corr)

print(f"Cross-validated Pearson Correlations: {pearson_scores}")
print(f"Mean Pearson Correlation: {np.mean(pearson_scores)}")


import joblib

# 모델 저장 (여기서 'best_meta_model.pkl'은 모델이 저장될 파일 이름)
# joblib.dump(meta_model_final, 'best_meta_model.pkl')

# 최종적으로 전체 데이터로 메타 모델 훈련
meta_model_final = best_meta_model
meta_model_final.fit(X_scaled, y)

# 테스트 데이터 예측
test_kykim_electra_df = pd.read_csv('output/STSModel_kykim-electra-kor-base_val_pearson=0.9258827567100525/test_output_kykim.csv')
test_deberta_df = pd.read_csv('output/STSModel_team-lucid-deberta-v3-base-korean_val_pearson=0.9326505064964294/test_output_lucid.csv')
test_snunlp_df = pd.read_csv('output/STSModel_snunlp-KR-ELECTRA-discriminator_val_pearson=0.9332568049430847/test_output_snunlp.csv')
test_eng2_df = pd.read_csv('output/eng2_/test_output_eng2.csv')

# 필요한 경우 target 열의 값에서 숫자만 추출
# test_deberta_df['target'] = test_deberta_df['target'].apply(extract_number)
# test_snunlp_df['target'] = test_snunlp_df['target'].apply(extract_number)

# 테스트 데이터의 예측값 병합
test_stacked_df = pd.DataFrame({
    'kykim_preds': test_kykim_electra_df['target'],
    'deberta_preds': test_deberta_df['target'],
    'snunlp_preds': test_snunlp_df['target'],
    'eng2_preds': test_eng2_df['target']
})

# 테스트 데이터 스케일링
test_X_scaled = scaler.transform(test_stacked_df)

# 저장된 모델을 불러오기
# loaded_meta_model = joblib.load('best_meta_model.pkl')

# 테스트 데이터 예측
test_predictions = meta_model_final.predict(test_X_scaled)

# 결과 저장
output_df = pd.DataFrame({
    'target': test_predictions
})
output_df.to_csv('output/Ensemble/blending_ensemble.csv', index=False)
