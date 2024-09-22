import torch
from tqdm import tqdm
import base.base_data_loader as module_data
import module.loss as module_loss
import module.metric as module_metric
import module.model as module_arch
import pandas as pd
import os

from utils import *
"""
원본 코드
def inference(dataloader, model, criterion, metrics, device):
    outputs, targets = [], []
    total_loss = 0
    with torch.no_grad():
        for (data, target) in tqdm(dataloader):
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            total_loss += loss.item()
            outputs.append(output)
            targets.append(target)

    outputs = torch.cat(outputs).squeeze()
    targets = torch.cat(targets).squeeze()
    result = {}
    result["loss"] = total_loss/len(dataloader)
    for metric in metrics:
        result[f"{metric.__name__}"] = metric(outputs, targets)

    return result, outputs
"""
def inference(dataloader, model, criterion, metrics, device):
    model.eval()  # 모델을 평가 모드로 설정
    total_loss = 0
    outputs, targets = [], []

    with torch.no_grad():  # Gradient 계산을 비활성화
        for batch in dataloader:
            # batch가 tuple인지 확인하여 data와 target을 분리
            if isinstance(batch, tuple):
                data = batch[0]
                target = batch[1] if len(batch) > 1 else None  # test 데이터는 target이 없을 수 있음
            else:
                data = batch
                target = None  # test 데이터이므로 target이 없음

            data = data.to(device)  # data를 device로 전송

            # 모델 예측
            output = model(data)
            outputs.append(output)

            # target이 있을 경우에만 손실을 계산
            if target is not None:
                target = target.to(device)
                loss = criterion(output, target)
                total_loss += loss.item()
                targets.append(target)

    outputs = torch.cat(outputs)  # 모든 배치의 출력값을 결합

    # target이 있을 때만 결과를 처리
    if targets:
        targets = torch.cat(targets)  # 모든 배치의 타겟을 결합
        result = {"test_loss": total_loss / len(dataloader)}
        for metric in metrics:
            result[f"test_{metric.__name__}"] = metric(outputs, targets)
    else:
        result = {"test_loss": None}  # test 데이터에서는 loss가 없으므로 None으로 설정

    return result, outputs



def main(checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    config = checkpoint['config']
    config['data_module']['args']['shuffle'] = False

    # 1. set data_module(=pl.DataModule class)
    data_module = init_obj(config['data_module']['type'], config['data_module']['args'], module_data)
    train_dataloader = data_module.train_dataloader()
    test_dataloader = data_module.test_dataloader()
    predict_dataloader = data_module.predict_dataloader()

    # 2. set model(=nn.Module class)
    model = init_obj(config['arch']['type'], config['arch']['args'], module_arch)
    model.load_state_dict(checkpoint['model_state_dict'])

    # 3. set deivce(cpu or gpu)
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    model = model.to(device)
    model.eval()

    # 4. set loss function & matrics 
    criterion = getattr(module_loss, config['loss'])
    metrics = [getattr(module_metric, met) for met in config['metrics']]

    """
    원본 코드
    # 5. inference
    train_result, train_outputs = inference(train_dataloader, model, criterion, metrics, device)
    test_result, test_outputs = inference(test_dataloader, model, criterion, metrics, device)
    print(train_result)
    print(test_result)
  
    # 6. save output
    pwd = os.getcwd()
    if not os.path.exists(f'{pwd}/output/'):
        os.makedirs(f'{pwd}/output/')
    folder_name = checkpoint_path.split("/")[-1].replace(".pth", "")
    folder_path = f'{pwd}/output/{folder_name}'
    os.makedirs(folder_path)

    train_df = pd.read_csv(config["data_module"]["args"]["train_path"])
    dev_df = pd.read_csv(config["data_module"]["args"]["test_path"])
    train_df['target'] = train_outputs.tolist()
    dev_df['target'] = test_outputs.tolist()
    train_df.to_csv(f'{folder_path}/train_output.csv', index=False)
    dev_df.to_csv(f'{folder_path}/dev_output.csv', index=False)

    outputs = []
    with torch.no_grad():
        for data in tqdm(predict_dataloader):
            data = data.to(device)
            output = model(data)
            outputs.append(output)

    outputs = torch.cat(outputs).squeeze()
    test_df = pd.read_csv(f'{pwd}/data/sample_submission.csv')
    test_df['target'] = outputs.tolist()
    test_df.to_csv(f'{folder_path}/test_output.csv', index=False)
    """
    # 5. inference
    train_result, train_outputs = inference(train_dataloader, model, criterion, metrics, device)
    test_result, test_outputs = inference(test_dataloader, model, criterion, metrics, device)
    print(train_result)
    print(test_result)
  
    # 6. save output
    pwd = os.getcwd()
    if not os.path.exists(f'{pwd}/output/'):
        os.makedirs(f'{pwd}/output/')
    folder_name = checkpoint_path.split("/")[-1].replace(".pth", "")
    folder_path = f'{pwd}/output/{folder_name}'
    os.makedirs(folder_path)

    train_df = pd.read_csv(config["data_module"]["args"]["train_path"])
    dev_df = pd.read_csv(config["data_module"]["args"]["test_path"])
    train_df['target'] = train_outputs.tolist()
    dev_df['target'] = test_outputs.tolist()
    train_df.to_csv(f'{folder_path}/train_output.csv', index=False)
    dev_df.to_csv(f'{folder_path}/dev_output.csv', index=False)

    outputs = []
    with torch.no_grad():
        for batch in tqdm(predict_dataloader):
            # batch가 tuple인지 확인하여 data를 추출
            if isinstance(batch, tuple):
                data = batch[0]  # 첫 번째 요소가 입력 데이터라고 가정
            else:
                data = batch

            data = data.to(device)  # 입력 데이터를 device로 전송

            # 모델 예측
            output = model(data)
            outputs.append(output)

    outputs = torch.cat(outputs).squeeze()

    test_df = pd.read_csv(f'{pwd}/data2/raw/sample_submission.csv')
    test_df['target'] = outputs.tolist()
    test_df.to_csv(f'{folder_path}/test_output.csv', index=False)

if __name__ == '__main__':

    checkpoint_path = "/data/ephemeral/home/level1-semantictextsimilarity-nlp-13/pytorch-template/saved/STSModel_team-lucid-deberta-v3-base-korean_val_pearson=0.9170345067977905.pth"
    main(checkpoint_path)
