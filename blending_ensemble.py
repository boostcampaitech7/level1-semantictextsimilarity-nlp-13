import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr
from itertools import combinations
import numpy as np
import os

class BlendingEnsemble:
    def __init__(self, train_file_paths, test_file_paths):
        """
        초기화 함수. 파일 경로들을 받아서 모델 학습과 예측을 위한 준비를 합니다.
        """
        self.train_file_paths = train_file_paths
        self.test_file_paths = test_file_paths
        self.scaler = StandardScaler()
        self.meta_model_final = None
    
    def extract_model_name(self, file_path):
        """
        파일 경로에서 모델 이름을 추출.
        첫 번째 '_'가 나올 때까지의 부분을 모델 이름으로 추출.
        """
        file_name = os.path.basename(file_path)
        model_name = file_name.split('_')[0]
        return model_name

    def load_predictions(self, file_path, output_type='dev'):
        """
        파일에서 예측값을 로드하고, 'target' 열이 문자열이면 숫자형으로 변환하여 반환.
        output_type이 'dev'면 dev_output.csv를, 'test'면 test_output.csv를 로드.
        """
        file_name = f'{output_type}_output.csv'
        try:
            df = pd.read_csv(os.path.join(file_path, file_name), encoding='utf-8')
        except UnicodeDecodeError:
            df = pd.read_csv(os.path.join(file_path, file_name), encoding='ISO-8859-1')
        
        # 'target' 열이 문자열일 경우 숫자로 변환하는 함수
        def extract_number(text):
            clean_text = str(text).replace('[', '').replace(']', '').strip()
            return float(clean_text)
        
        if df['target'].dtype == object:  # 문자열일 경우
            df['target'] = df['target'].apply(extract_number)
        
        return df

    def create_stacked_df(self, file_paths, output_type='dev'):
        """
        입력받은 파일 경로 리스트에 따라 모델 예측값으로 구성된 stacked_df 생성.
        파일 경로의 수에 관계없이 처리 가능. output_type에 따라 'dev' 또는 'test' 데이터를 로드.
        """
        stacked_df = pd.DataFrame()
        
        for file_path in file_paths:
            model_name = self.extract_model_name(file_path)
            df = self.load_predictions(file_path, output_type=output_type)
            
            # 모델 이름에 맞춰 예측값을 'model_name_preds'로 저장
            stacked_df[f'{model_name}_preds'] = df['target']
        
        if output_type == 'dev':
            # dev 데이터일 경우 마지막 파일을 기준으로 라벨 설정 (모든 파일에서 label이 같다는 가정)
            stacked_df['label'] = df['label']  # 'label' 컬럼은 마지막 로드된 파일의 것
        
        return stacked_df

    def fit(self, save_model=False, model_path="best_meta_model.pkl"):
        """
        블랜딩 앙상블을 학습하는 함수. 파일 경로 리스트로부터 모델 예측값을 로드하여 메타 모델을 학습하고 평가.
        """
        # Stacked DataFrame 생성 (훈련 데이터)
        stacked_df = self.create_stacked_df(self.train_file_paths, output_type='dev')
        
        # 메타 모델의 입력과 타겟 변수 설정
        X = stacked_df.drop(columns=['label'])  # 예측값들을 입력으로 사용
        y = stacked_df['label']
        
        # 예측값 스케일링
        X_scaled = self.scaler.fit_transform(X)
        
        # 교차 검증 설정
        kf = KFold(n_splits=5, shuffle=True, random_state=10)
        meta_predictions = np.zeros(len(X))  # 메타 모델의 예측값을 저장할 배열
        pearson_scores = []
        
        # Ridge 회귀 모델 하이퍼파라미터 튜닝을 위한 그리드
        param_grid = {
            'alpha': [0.01, 0.1, 1.0, 10.0, 100.0]
        }
        
        # 교차 검증을 통해 메타 모델 훈련 및 평가
        for train_index, test_index in kf.split(X_scaled):
            X_train, X_test = X_scaled[train_index], X_scaled[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]
            
            # 메타 모델로 Ridge 사용
            grid_search = GridSearchCV(Ridge(), param_grid, cv=5)
            grid_search.fit(X_train, y_train)
            best_meta_model = grid_search.best_estimator_
            
            y_pred = best_meta_model.predict(X_test)
            meta_predictions[test_index] = y_pred
            
            pearson_corr, _ = pearsonr(y_pred, y_test)
            pearson_scores.append(pearson_corr)

        # # 평균 Pearson Correlation 계산
        # mean_pearson_corr = np.mean(pearson_scores)
        # print(f"Mean Pearson Correlation: {mean_pearson_corr}")
        
        # 최종적으로 전체 데이터로 메타 모델 훈련
        self.meta_model_final = best_meta_model
        self.meta_model_final.fit(X_scaled, y)

        return mean_pearson_corr

    def predict(self):
        """
        테스트 데이터를 기반으로 예측을 수행하는 함수.
        """
        # 테스트 데이터를 로드하여 stacked_df 생성
        test_stacked_df = self.create_stacked_df(self.test_file_paths, output_type='test')
        
        # 테스트 데이터 스케일링
        test_X_scaled = self.scaler.transform(test_stacked_df)
        
        # 테스트 데이터 예측
        test_predictions = self.meta_model_final.predict(test_X_scaled)
        
        # 결과 반환
        return test_predictions

DebertaTW_path = 'output\STSModel_team-lucid-deberta-v3-base-korean_val_pearson=0.9326505064964294'
KykimTW_path = 'output\STSModel_kykim-electra-kor-base_val_pearson=0.9258827567100525'
SnuTW_path = 'output\STSModel_snunlp-KR-ELECTRA-discriminator_val_pearson=0.9332568049430847'
eng2_path = 'output\STSModel_team-lucid-deberta-v3-base-korean_val_pearson=0.9326505064964294'

# 모델 경로들
model_paths = [
    DebertaTW_path, KykimTW_path, SnuTW_path, eng2_path
]

# 결과 저장
results = []


model_names = [BlendingEnsemble([], []).extract_model_name(path) for path in combo]
blending_ensemble = BlendingEnsemble(combo, combo)  # 학습 및 테스트 데이터 동일하게 사용
pearson_score = blending_ensemble.fit()
results.append((model_names, pearson_score))

# 결과를 DataFrame으로 변환
df_results = pd.DataFrame(results, columns=["Combination", "Pearson Score"])

# 모델 경로들
model_paths = [
    DebertaTW_path, KykimTW_path, SnuTW_path, eng2_path
]

# 테스트 경로들 - 경로는 학습과 일치해야 함
test_file_paths_map = {
    DebertaTW_path: 'output/DebertaTW_team-lucid-deberta-v3-base-korean_val_pearson=0.9326505064964294',
    KykimTW_path: 'output/KykimTW_kykim-electra-kor-base_val_pearson=0.9258827567100525',
    SnuTW_path: 'output/SnunlpTW_snunlp-KR-ELECTRA-discriminator_val_pearson=0.9332568049430847',
    eng2_path: 'output/eng2'
}

# 상위 4개의 조합
top_combinations = [
    [DebertaHS_path, KykimTW_path, SnuTW_path, eng2_path],
]

test = pd.read_csv('output/SnunlpTW_snunlp-KR-ELECTRA-discriminator_val_pearson=0.9332568049430847/test_output.csv')

# 결과 저장
for idx, combo in enumerate(top_combinations, start=1):
    # 해당 조합의 테스트 파일 경로 매핑
    test_combo_paths = [test_file_paths_map[path] for path in combo]
    
    print(f"\nEvaluating combination {idx}: {[BlendingEnsemble([], []).extract_model_name(path) for path in combo]}")
    
    # 학습 및 예측
    blending_ensemble = BlendingEnsemble(combo, test_combo_paths)  # 학습 조합에 맞는 테스트 경로 사용
    blending_ensemble.fit()  # 모델 학습
    test_predictions = blending_ensemble.predict()  # 테스트 데이터 예측
    
    # 결과를 CSV 파일로 저장
    output_df = pd.DataFrame({
        'id': test['id'],
        'target': test_predictions
    })

    output_file_path = f'output/Ensemble/blending_ensemble.csv'
    output_df.to_csv(output_file_path, index=False)


