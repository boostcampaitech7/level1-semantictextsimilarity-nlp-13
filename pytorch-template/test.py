import torch
from tqdm import tqdm
import base.base_data_loader as module_data
import module.loss as module_loss
import module.metric as module_metric
import module.model as module_arch
import pandas as pd
import os

from utils import *

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

if __name__ == '__main__':

    checkpoint_path = "/data/ephemeral/home/gj/pytorch-template/saved/STSModel_snunlp-KR-ELECTRA-discriminator_val_pearson=0.9253344535827637.pth"
    main(checkpoint_path)
