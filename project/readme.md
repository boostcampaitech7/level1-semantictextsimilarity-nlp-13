## 훈련 ##

0. baseline_config.yaml을 원하는 세팅으로 수정

1. train.py 실행하여 모델을 학습시킵니다
    ├─ data_pipeline.py: Dataloader class 제공
    └─ model.py : Model class 제공
    
2. experiments 폴더에 학습된 모델이 저장됩니다

3. 실험 종료 후 val_pearson 값을 확인하고
   experiments/README.md에 결과를 기록합니다


## 예측 ## 

1. inference.py에서 불러올 모델을 세팅합니다

2. inference.py를 실행하여 output.csv를 출력

3. 리더보드에 제출 후 결과 분석 시작


## 분석 ## 

1. 자기만의 또는 팀만의 스타일로 분석
ex) Binary-label로 Confusion Matrix 만들기
    특별히 정확도가 떨어지는 label 값이 있나 확인

2. 분석 결과에 따라 모델 피드백
ex) tokenizer 교체 / hyperparameter 수정
    데이터 증강 / sentence_1,2에 개별적인 모델 적용
    다른 모델로 교체 / low-level에서 모델 커스텀

3. 다시 훈련 후 평가

## base line
baseline config 파일에 한국어 주석 달면 오류남