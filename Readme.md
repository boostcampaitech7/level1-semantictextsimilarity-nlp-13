# Level 1 Project :: STS(Semantic Text Similarity)

### 📝 Abstract
- 이 프로젝트는 네이버 부스트 캠프 AI-Tech 7기 NLP Level 1 기초 프로젝트 경진대회로, Dacon과 Kaggle과 유사한 대회형 방식으로 진행되었다.
- 두 문장이 의미적으로 얼마나 유사한지를 수치화하는 N21 자연어처리 Task인 의미 유사도 판별(Semantic Text Similarity, STS)를 주제로, 모든 팀원이 데이터 전처리부터 앙상블까지 AI 모델링의 전 과정을 함께 협업했다.

<br>

## Project Leader Board 
- Public Leader Board
<img width="450" alt="public_leader_board" src="https://github.com/user-attachments/assets/4d4592bc-1e3e-4455-8de9-cf61e5fc6d50">

- Private Leader Board 
<img width="450" alt="private_leader_board" src="https://github.com/user-attachments/assets/f1a5b53d-f30b-4d87-8a14-3cc1a602f8a0">

- [📈 NLP 13조 Project Wrap-Up report 살펴보기](https://github.com/user-attachments/files/17182231/NLP_13.Wrap-Up.pdf
)

<br>

## 🧑🏻‍💻 Team Introduction & Members 

> Team name : 스빈라킨스배 [ NLP 13조 ]

### 👨🏼‍💻 Members
권지수|김성은|김태원|이한서|정주현|
:-:|:-:|:-:|:-:|:-:
<img src='https://github.com/user-attachments/assets/ab4b7189-ec53-41be-8569-f40619b596ce' height=125 width=100></img>|<img src='https://github.com/user-attachments/assets/49dc0e59-93ee-4e08-9126-4a3deca9d530' height=125 width=100></img>|<img src='https://github.com/user-attachments/assets/a15b0f0b-cd89-412b-9b3d-f59eb9787613' height=125 width=100></img>|<img src='https://github.com/user-attachments/assets/11b2ed88-bf94-4741-9df5-5eb2b9641a9b' height=125 width=100></img>|<img src='https://github.com/user-attachments/assets/3e2d2a7e-1c64-4cb7-97f6-a2865de0c594' height=125 width=100></img>
[Github](https://github.com/Kwon-Jisu)|[Github](https://github.com/ssungni)|[Github](https://github.com/chris40461)|[Github](https://github.com/beaver-zip)|[Github](https://github.com/peter520416)
<a href="mailto:wltn80609@ajou.ac.kr" target="_blank"><img src="https://img.shields.io/badge/Gmail-EA4335?style&logo=Gmail&logoColor=white"/></a>|<a href="mailto:sunny020111@ajou.ac.kr" target="_blank"><img src="https://img.shields.io/badge/Gmail-EA4335?style&logo=Gmail&logoColor=white"/></a>|<a href="mailto:chris40461@gmail.com" target="_blank"><img src="https://img.shields.io/badge/Gmail-EA4335?style&logo=Gmail&logoColor=white"/></a>|<a href="mailto:beaver.zip@gmail.com" target="_blank"><img src="https://img.shields.io/badge/Gmail-EA4335?style&logo=Gmail&logoColor=white"/></a>|<a href="mailto:peter520416@gmail.com" target="_blank"><img src="https://img.shields.io/badge/Gmail-EA4335?style&logo=Gmail&logoColor=white"/></a>|

### 🧑🏻‍🔧 Members' Role
> 김태원 캠퍼는 13조의 팀장을 맡아 팀을 이끌며, 데이터의 분포를 분석하고, 데이터 증강에 큰 기여를 했다. 정주현 캠퍼와 이한서 캠퍼는 주로 정규 표현식을 활용하여 데이터를 전처리하고, 조사(은, 는, 이, 가, 을, 를, 과, 와)를 랜덤화하는 기법으로 데이터 품질을 향상시켰다. 김성은 캠퍼와 권지수 캠퍼는 Hugging Face와 GitHub에 있는 다양한 모델을 탐색하며, 최적화 작업을 통해 성능을 극대화하는데 주력했다.

| 이름 | 역할 |
| :---: | --- |
| **`권지수`** | **EDA** (라벨 분포 데이터분석), **모델 탐색** (KLUE: 논문 바탕으로 RoBERTa와 ELECTRA 계열 모델 중심으로 탐색), **모델 실험** (team-lucid/deberta-v3-base-korean), **Ensemble 실험** (output 평균 및 가중치 활용) |
| **`김성은`** | **EDA** (라벨 분포 데이터분석), **모델 탐색** (Encoder, Decoder, Encoder - Decoder 모델로 세분화하여 탐색), **모델 실험** (snunlp-KR-ELECTRA), **Ensemble 실험** (output 평균 및 가중치 활용) |
| **`김태원`** | **모델 실험** (KR-ELECTRA-discriminator, electra-kor-base, deberta-v3, klue-roberta ), **데이터 증강** (label rescaling(0점 인덱스의 제거 및 5점 인덱스 추가), 단순 복제 데이터 증강(1점~3점 인덱스), train 데이터의 전체적인 맞춤법 교정/불용어 제거/띄어쓰기 교정), **모델 Ensemble** (weighted sum for 3model/4models) |
| **`이한서`** |**데이터 증강**(조사 대체, Label 분포 균형화), **모델 실험**(team-lucid/deberta-v3-base-korean, monologg/koelectra-base-v3-discriminator, snunlp/KR-ELECTRA), Hyperparameter Tuning(Optuna Template 제작 및 실험)|
| **`정주현`** &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; | **데이터 EDA** (Label 분포, 문장 내의 단어 빈도), **데이터 증강** (Swap sentence1 and sentence2, 유의어 교체(‘너무’, ‘진짜’, ‘정말’)), **모델 선정 및 Ensemble** (T5-base-korean-summarization), Ensemble(Blending Ensemble for 3 or 4 model(meta model = Ridge)) |

<br>

## 🖥️ Project Introduction 


|**프로젝트 주제**| **`Semantic Text Similarity (STS)` :** 두 텍스트가 얼마나 유사한지 판단하는 NLP Task|
| :---: | --- |
|**프로젝트 구현내용**| 1. Hugging Face 의 Pretrained 모델과 STS 데이터셋을 활용해 두 문장의 0 과 5 사이의 유사도를 측정하는 AI 모델을 구축 <br>2. 리더보드 평가지표인 피어슨 상관 계수(Pearson Correlation Coefficient ,PCC)에서 높은 점수(1에 가까운 점수)에 도달할 수 있도록 데이터 전처리, 증강, 하이퍼 파라미터 튜닝을 진행|
|**개발 환경**|**• `GPU` :** Tesla V100 서버 4개 (RAM32G)<br> **• `개발 Tool` :** Jupyter notebook, VS Code [서버 SSH연결]
|**협업 환경**|**• `Github Repository` :** Baseline 코드 공유 및 버전 관리, 개인 branch를 사용해 작업상황 공유 <br>**• `Notion` :** STS 프로젝트 페이지를 통한 역할분담, 실험 가설 설정 및 결과 공유 <br>**• `SLACK, Zoom` :** 실시간 대면/비대면 회의|

<br>

## 📁 Project Structure

### 🗂️ 디렉토리 구조 설명
- 학습 데이터 경로: `./data`
- 학습 메인 코드: `./train.py`
- 학습 데이터셋 경로: `./data/aug_train.csv`
- 테스트 데이터셋 경로: `./data/test.csv`

### 📄 코드 구조 설명

> 학습 진행하기 전 데이터 증강을 먼저 실행하여 학습 시간 단축

- **데이터 증강 Get Augmentation Data** : `augmentation.py`
- **Train** : `train.py`
- **Predict** : `test.py`
- **Ensemble** : `weighted_ensemble.py`, `blending_ensemble.py`
- **최종 제출 파일** : `/output/Ensemble/blending_ensemble.csv`

```
📦level1_semantictextsimilarity-nlp-13
 ┣ 📂 base
 ┃ ┣ __init__.py
 ┃ ┣ base_data_loader.py
 ┃ ┣ base_dataset.py
 ┃ ┗ base_trainer.py
 ┣ 📂 data
 ┃ ┣ aug_train.csv
 ┃ ┣ dev.csv
 ┃ ┣ sample_submission.csv
 ┃ ┣ test.csv
 ┃ ┗ train.csv
 ┣ 📂 module
 ┃ ┣ dataset.py
 ┃ ┣ loss.py
 ┃ ┣ metric.py
 ┃ ┣ model.py
 ┃ ┗ trainer.py
 ┣ 📂 output
 ┃ ┣ 📂 Ensemble
 ┃ ┃ ┗ blending_ensemble.csv
 ┃ ┣ 📂 STSModel_eenzeenee-t5-base-korean-summarization
 ┃ ┃ ┣ dev_output_t5.csv
 ┃ ┃ ┣ test_output_t5.csv
 ┃ ┃ ┗ train_output_t5.csv
 ┃ ┣ 📂 STSModel_klue-roberta-base
 ┃ ┃ ┣ dev_output_klue.csv
 ┃ ┃ ┣ test_output_klue.csv
 ┃ ┃ ┗ train_output.csv
 ┃ ┣ 📂 STSModel_kykim-electra-kor-base
 ┃ ┃ ┣ dev_output_kykim.csv
 ┃ ┃ ┣ test_output_kykim.csv
 ┃ ┃ ┗ train_output.csv
 ┃ ┣ 📂 STSModel_snunlp-KR-ELECTRA-discriminator
 ┃ ┃ ┣ dev_output_snunlp.csv
 ┃ ┃ ┣ test_output_snunlp.csv
 ┃ ┃ ┗ train_output.csv
 ┃ ┣ 📂 STSModel_team-lucid-deberta-v3-base-korean
 ┃ ┃ ┣ dev_output_lucid.csv
 ┃ ┃ ┣ test_output_lucid.csv
 ┃ ┃ ┗ train_output.csv
 ┃ ┗ 📂 eng2_
 ┃ ┃ ┣ dev_output_eng2.csv
 ┃ ┃ ┗ test_output_eng2.csv
 ┣ augmentation.py
 ┣ blending_ensemble.py
 ┣ config.yaml
 ┣ Readme.md
 ┣ requirements.txt
 ┣ test.py
 ┣ train.py
 ┗ utils.py
 ```
<br>

## 📐 Project Ground Rule
>팀 협업을 위해 프로젝트 관련 Ground Rule을 설정하여 프로젝트가 원활하게 돌아갈 수 있도록 규칙을 정했으며, 날짜 단위로 간략한 목표를 설정하여 협업을 원활하게 진행할 수 있도록 계획하여 진행했다.

**- a. `Server 관련`** : 권지수, 김성은, 이한서, 정주현 캠퍼는 각자 서버를 생성해 모델 실험을 진행하고, 김태원 캠퍼는 서버가 유휴 상태인 서버에서 실험을 진행했다.

**- b. `Git 관련`** : 각자 branch 생성해 작업하고, 공통으로 사용할 파일은 main에 push 하는 방법으로 협업했다.

**- c. `Submission 관련`** : 대회 마감 2일 전까지는 자유롭게 제출했고, 2일 전부터는 인당 2회씩 분배했다.

**- d. `Notion 관련`** : 원활한 아이디어 브레인스토밍과 분업을 위해 회의를 할 경우 노션에 기록하며, 연구 및 실험결과의 기록을 공유했다.

<br>

## 🗓 Project Procedure: 총 14일 진행
- **`(1~3 일차)`**: EDA 분석
- **`(3~5 일차)`**: 데이터 전처리
- **`(6~11 일차)`** : 데이터 증강
- **`(7~12 일차)`** : 모델링 및 튜닝
- **`(11~13 일차)`** : 앙상블

* 아래는 저희 프로젝트 진행과정을 담은 Gantt차트 입니다. 
<img width="800" alt="Gantt" src="https://github.com/user-attachments/assets/23c3dacb-95c7-49c0-88df-ebc5b5f9b1a7">

<br>

## **📊DataSet**
* 우리는 먼저 데이터의 양이 적고 불균형하다는 점을 확인했다. 이를 해결하기 위해 데이터의 양을 절대적으로 늘린 후, 증강된 데이터의 라벨 분포를 고려하여 추가적인 데이터 증강을 진행했다.

|**Techniques**|**Description**|
|:--:|:--:|
|**Swap sentence**|전체 데이터를 sentence1과 sentence2의 순서를 바꾸어 데이터의 양을 약 2배로 증강했다.|
|**유의어 교체를 통한 증강**|raw 데이터셋에서 sentence 1과 sentence 2에 '너무', '정말', '진짜'라는 단어가 많이 들어가 있다는 것을 판단하여 해당 단어들이 포함된 문장에서 그 중 한 단어를 제외한 나머지 두 단어 중 하나로 무작위 대체하여 데이터를 증강했다.|

<br>



## 🤖**Ensemble**

* 최종적으로 5개의 모델에 대해 Blending Ensemble을 수행했다.

|**Model**|**Learing Rate**|**Batch Size**|**loss**|**epoch**|**dev person (val_pearson)**|**Scheduler**|
|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
|**klue/roberta-base**|1e-5|16|L2(MSE)|10|0.9216|stepLR|
|**kykim/electra-kor-base**|1e-5|16|L2(MSE)|10|0.9259|stepLR|
|**team-lucid/deberta-v3-base-korean**|1e-4|16|L2(MSE)|10|0.9327|stepLR|
|**snunlp/KR-ELECTRA-discriminator**|1e-5|16|L2(MSE)|10|0.9333|stepLR|
|**eenzeenee/t5-base-korean-summarization**|1e-5|16|L2(MSE)|10|0.9229|stepLR|

<br>

## 💻 Getting Started

### ⚠️  How To install Requirements
```bash
# 필요 라이브러리 설치
pip install -r requirements.txt
```

### ⌨️ How To Train & Test
```bash
#데이터 증강
python3 augmentation.py
# train.py 코드 실행 : 모델 학습을 순차적으로 진행
# 이후 test.py 코드를 순차적으로 실행하여 test
# config.yaml 내 모델 이름, lr 을 리스트 순서대로 변경하며 train 으로 학습

#plm_name[0], lr[0] -> klue/roberta-base
python3 train.py
python3 test.py

#plm_name[1], lr[1] -> kykim/electra-kor-base
python3 train.py
python3 test.py

#plm_name[2], lr[2] -> team-lucid/deberta-v3-base-korean 
python3 train.py
python3 test.py

#plm_name[3], lr[3] -> snunlp/KR-ELECTRA-discriminator
python3 train.py
python3 test.py

#plm_name[4], lr[4] -> eenzeenee/t5-base-korean- summarization
python3 train.py
python3 test.py

```

### ⌨️ How To Ensemble
```bash
# 순차적으로 weighted ensemble 진행 후, 출력 결과를 사용해서 blended ensemble 진행
python3 weighted_ensemble.py # klue/roberta-base, eenzeenee/t5-base-korean-summarization, kykim/electra-kor-base
python3 blending_ensemble.py # kykim/electra-kor-base , team-lucid/deberta-v3-base-korean , snunlp/KR-ELECTRA-discriminator, weighted_ensemble
```

