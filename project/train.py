import argparse
import yaml
import os

import torch
#import transformers
#import pandas as pd

import pytorch_lightning as pl
#import wandb
##############################
from utils import data_pipeline, utils
from model.model import Model
##############################


if __name__ == "__main__":
    # dataset path 설정
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_path', default='./data/raw/train.csv')
    parser.add_argument('--dev_path', default='./data/raw/dev.csv')
    parser.add_argument('--test_path', default='./data/raw/dev.csv')
    ## test_path도 dev.csv로 설정 >> trainer.test()에서 dev.csv 사용
    parser.add_argument('--predict_path', default='./data/raw/test.csv')
    ## predict_path에 test.csv로 설정 >> trainer.predict()에서 test.csv 사용
    args = parser.parse_args(args=[])

    # baseline_config 설정 불러오기
    with open('baselines/baseline_config.yaml') as f:
        CFG = yaml.load(f, Loader=yaml.FullLoader)
    
    # experiments 폴더 내부에 실험 폴더 생성
    # 폴더 이름 : 실험 날짜 - 실험 시간 - admin
    experiment_path = utils.create_experiment_folder(CFG)


    # dataloader / model 설정
    dataloader = data_pipeline.Dataloader(CFG, args.train_path, args.dev_path, args.test_path, args.predict_path)
    model = Model(CFG)

    # trainer 인스턴스 생성
    trainer = pl.Trainer(accelerator="cpu", devices=1, max_epochs=CFG['train']['epoch'], log_every_n_steps=1)
    # accelerator="gpu"
    # Train part
    trainer.fit(model=model, datamodule=dataloader)
    ## datamodule에서 train_dataloader와 val_dataloader를 호출
    
    ## Dataloader 내부에 val_dataloader 부분을 수정해서
    ## valid set을 바꿀 수 있음

    trainer.test(model=model, datamodule=dataloader)
    ## datamodule에서 test_dataloader 호출
    ## predict_path로 설정된 test.csv가 사용된다

    # 학습된 모델 저장 (experiment_folder 안에 model.pt로 저장)
    torch.save(model, os.path.join(experiment_path, 'model.pt'))
    print(f"모델이 저장되었습니다: {experiment_path}")