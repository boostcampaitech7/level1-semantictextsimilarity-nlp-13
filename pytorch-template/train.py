'''
import os
import torch
import numpy as np
import base.base_data_loader as module_data
import module.loss as module_loss
import module.metric as module_metric
import module.model as module_arch
from module.trainer import Trainer
from omegaconf import DictConfig, OmegaConf
from utils import *
import hydra

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# fix random seeds for reproducibility
SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)


@hydra.main(config_path=".", config_name="config", version_base=None)
def main(cfg):

    # 0. DictConfig to dict
    cfg.pwd = os.getcwd()
    config = OmegaConf.to_container(cfg, resolve=True)

    # 1. set data_module(=pl.DataModule class)
    data_module = init_obj(
        config["data_module"]["type"], config["data_module"]["args"], module_data
    )

    # 2. set model(=nn.Module class)
    model = init_obj(config["arch"]["type"], config["arch"]["args"], module_arch)

    # 3. set deivce(cpu or gpu)
    # 장치 설정
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    model = model.to(device)

    # 4. set loss function & matrics
    criterion = getattr(module_loss, config["loss"])
    metrics = [getattr(module_metric, met) for met in config["metrics"]]

    # 5. set optimizer & learning scheduler
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = init_obj(
        config["optimizer"]["type"],
        config["optimizer"]["args"],
        torch.optim,
        trainable_params,
    )
    lr_scheduler = init_obj(
        config["lr_scheduler"]["type"],
        config["lr_scheduler"]["args"],
        torch.optim.lr_scheduler,
        optimizer,
    )

    # 6. 위에서 설정한 내용들을 trainer에 넣는다.
    trainer = Trainer(
        model,
        criterion,
        metrics,
        optimizer,
        config=config,
        device=device,
        data_module=data_module,
        lr_scheduler=lr_scheduler,
    )

    # 6. train
    trainer.train()


if __name__ == "__main__":
    main()
'''

# import os
# import torch
# import numpy as np
# import base.base_data_loader as module_data
# import module.loss as module_loss
# import module.metric as module_metric
# import module.model as module_arch
# from module.trainer import Trainer
# from omegaconf import DictConfig, OmegaConf
# from utils import *
# import hydra



# os.environ["TOKENIZERS_PARALLELISM"] = "false"

# # fix random seeds for reproducibility
# SEED = 123
# torch.manual_seed(SEED)
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False
# np.random.seed(SEED)


# @hydra.main(config_path=".", config_name="config", version_base=None)
# def main(cfg):

#     # 0. DictConfig to dict
#     cfg.pwd = os.getcwd()
#     config = OmegaConf.to_container(cfg, resolve=True)

#     # 1. set data_module(=pl.DataModule class)
#     data_module = init_obj(
#         config["data_module"]["type"], config["data_module"]["args"], module_data
#     )

#     # 2. set model(=nn.Module class)
#     model = init_obj(config["arch"]["type"], config["arch"]["args"], module_arch)

#     # 3. set deivce(cpu or gpu)
#     # 장치 설정
#     if torch.cuda.is_available():
#         device = torch.device("cuda")
#     elif torch.backends.mps.is_available():
#         device = torch.device("mps")
#     else:
#         device = torch.device("cpu")
#     model = model.to(device)

#     # 4. set loss function & matrics
#     criterion = getattr(module_loss, config["loss"])
#     metrics = [getattr(module_metric, met) for met in config["metrics"]]

#     # 5. set optimizer & learning scheduler
#     trainable_params = filter(lambda p: p.requires_grad, model.parameters())
#     optimizer = init_obj(
#         config["optimizer"]["type"],
#         config["optimizer"]["args"],
#         torch.optim,
#         trainable_params,
#     )
#     lr_scheduler = init_obj(
#         config["lr_scheduler"]["type"],
#         config["lr_scheduler"]["args"],
#         torch.optim.lr_scheduler,
#         optimizer,
#     )

#     # 6. 위에서 설정한 내용들을 trainer에 넣는다.
#     trainer = Trainer(
#         model,
#         criterion,
#         metrics,
#         optimizer,
#         config=config,
#         device=device,
#         data_module=data_module,
#         lr_scheduler=lr_scheduler,
#     )

#     # 6. train
#     trainer.train()


# if __name__ == "__main__":
#     main()



import os
import torch
import numpy as np
import base.base_data_loader as module_data
import module.loss as module_loss
import module.metric as module_metric
import module.model as module_arch
from module.trainer import Trainer
from omegaconf import DictConfig, OmegaConf
from utils import *
import hydra
import optuna
import pandas as pd  # CSV 파일 생성을 위해 추가

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# fix random seeds for reproducibility
SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)

# 각 trial의 epoch별 결과 저장
results_per_trial = []  
# Optuna의 objective 함수 정의
def objective(trial, cfg):  # cfg 인자를 추가
    # 하이퍼파라미터 설정
    lr = trial.suggest_float("lr", 5e-5, 1e-4, step=1e-5)  # log scale 없이 1e-5 단위로 탐색
    batch_size = 16  # 고정
    optimizer_type = "AdamW"  # 고정
    scheduler_type = trial.suggest_categorical("scheduler", ["CosineAnnealingLR", "StepLR"])
    loss_type = trial.suggest_categorical("loss_type", ["l1_loss", "l2_loss"])  # l1_loss, l2_loss 중 선택
    max_length = trial.suggest_categorical("max_length", [160, 224])  # 160, 224 중 선택
    weight_decay = 0.23  # 고정
    amsgrad = True  # 고정

    print(f"\nStarting experiment #{trial.number + 1}")
    print(f"Parameters: lr={lr}, batch_size={batch_size}, optimizer={optimizer_type}, "
          f"scheduler={scheduler_type}, loss={loss_type}, max_length={max_length}, "
          f"weight_decay={weight_decay}, amsgrad={amsgrad}")

    # config 업데이트
    config = OmegaConf.to_container(cfg, resolve=True)
    config["optimizer"]["type"] = optimizer_type
    config["optimizer"]["args"]["lr"] = lr
    config["optimizer"]["args"]["weight_decay"] = weight_decay
    config["data_module"]["args"]["batch_size"] = batch_size
    config["data_module"]["args"]["max_length"] = max_length
    config["loss"] = loss_type
    config["arch"]["args"]["dropout_rate"] = 0.3  # 드롭아웃 비율 설정
    config["optimizer"]["args"]["amsgrad"] = amsgrad

    # 학습률 스케줄러 설정 수정
    config["lr_scheduler"] = {}
    config["lr_scheduler"]["type"] = scheduler_type
    if scheduler_type == "CosineAnnealingLR":
        config["lr_scheduler"]["args"] = {
            "T_max": config["trainer"]["epochs"],
            "eta_min": 1e-6
        }
    elif scheduler_type == "StepLR":
        config["lr_scheduler"]["args"] = {
            "step_size": 5,
            "gamma": 0.5
        }
    else:
        config["lr_scheduler"]["args"] = {}  # 기타 스케줄러의 경우 기본값

    # 데이터 모듈, 모델 설정
    data_module = init_obj(config["data_module"]["type"], config["data_module"]["args"], module_data)
    model = init_obj(config["arch"]["type"], config["arch"]["args"], module_arch)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # 손실 함수 및 메트릭 설정
    criterion = getattr(module_loss, config["loss"])
    metrics = [getattr(module_metric, met) for met in config["metrics"]]

    # 옵티마이저 설정
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = init_obj(config["optimizer"]["type"], config["optimizer"]["args"], torch.optim, trainable_params)

    # 학습률 스케줄러 설정
    lr_scheduler = None
    if config.get("lr_scheduler"):
        lr_scheduler = init_obj(config["lr_scheduler"]["type"], config["lr_scheduler"]["args"],
                                torch.optim.lr_scheduler, optimizer)

    # Trainer에 설정된 내용 추가
    trainer = Trainer(model, criterion, metrics, optimizer, config=config, device=device,
                      data_module=data_module, lr_scheduler=lr_scheduler)

    # 조기 종료 조건 변수
    best_val_pearson = -float('inf')
    epochs_no_improve = 0

    # 각 trial의 결과를 저장할 딕셔너리
    trial_results = {
        'trial_number': trial.number, 
        'lr': lr, 
        'batch_size': batch_size,
        'scheduler_type': scheduler_type,  # scheduler_type 추가
        'loss_type': loss_type,  # loss_type 추가
        'max_length': max_length, 
        'weight_decay': weight_decay
    }
    epoch_val_pearsons = []

    # 학습 및 검증 진행
    for epoch in range(1, 11):
        print(f"Epoch {epoch}/{10} - Parameters: lr={lr}, batch_size={batch_size}, "
              f"optimizer={optimizer_type}, scheduler={scheduler_type}, loss={loss_type}, "
              f"max_length={max_length}, weight_decay={weight_decay}")
        trainer._train_epoch(epoch)

        val_pearson = trainer.best_score
        epoch_val_pearsons.append(val_pearson)
        print(f"val_pearson after epoch {epoch}: {val_pearson}")

        # 조기 종료 조건: 0.85 미만일 경우 종료
        if val_pearson < 0.85:
            print(f"val_pearson ({val_pearson}) is below threshold 0.85. Stopping this experiment early.")
            break

        # 성능 향상이 없을 경우 조기 종료 (최대 5 epoch 동안 향상이 없을 경우)
        if val_pearson > best_val_pearson:
            best_val_pearson = val_pearson
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= 5:
            print(f"No improvement in val_pearson for 5 epochs. Stopping early.")
            break

        # val_pearson이 0.93 이상일 경우 체크포인트 저장
        if val_pearson >= 0.93:
            trainer.save_checkpoint = True

    # 각 trial의 결과 기록
    trial_results.update({f'epoch_{i+1}_val_pearson': val for i, val in enumerate(epoch_val_pearsons)})
    results_per_trial.append(trial_results)

    return best_val_pearson


@hydra.main(config_path=".", config_name="config", version_base=None)
def main(cfg):
    try:
        # Optuna 스터디 생성 및 최적화 시작
        study = optuna.create_study(direction="maximize")  # Pearson correlation을 최대화
        study.optimize(lambda trial: objective(trial, cfg), n_trials=20)  # 실험 횟수를 20번으로 제한

    except Exception as e:
        print(f"Experiment interrupted due to an error: {e}")

    finally:
        # 각 trial의 epoch별 val_pearson 값 기록을 CSV 파일로 저장
        df_trial_results = pd.DataFrame(results_per_trial)
        df_trial_results.to_csv("trial_epoch_results.csv", index=False)
        print("Trial results saved to 'trial_epoch_results.csv'.")

if __name__ == "__main__":
    main()