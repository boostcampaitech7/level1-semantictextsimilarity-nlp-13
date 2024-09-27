import numpy as np
import torch
from base import BaseTrainer
from tqdm import tqdm
import wandb


class Trainer(BaseTrainer):
    """
    Trainer class
    """
    def __init__(self, model, criterion, metric_ftns, optimizer, config, device,
                 data_module, lr_scheduler=None):
        super().__init__(model, criterion, metric_ftns, optimizer, config)
        self.config = config
        self.device = device
        self.data_module = data_module
        self.lr_scheduler = lr_scheduler
        self.train_dataloader = self.data_module.train_dataloader()
        self.valid_dataloader = self.data_module.val_dataloader()

    def _train_epoch(self, epoch):
 
        self.model.train()
        self.len_epoch = len(self.train_dataloader)
        total_loss = 0
        for (data, target) in tqdm(self.train_dataloader, total=len(self.train_dataloader), desc=f"epoch {epoch} training"):
            # 1.1. prepare data
            data, target = data.to(self.device), target.to(self.device)
            # 1.2. initalize gradient by 0
            self.optimizer.zero_grad()
            # 1.3. run model
            output = self.model(data)
            # 1.4. calculate loss
            loss = self.criterion(output, target)
            total_loss = loss.item()
            # 1.5. calculate gradient
            loss.backward()
            # 1.6. optimize parameter
            self.optimizer.step()
    
            # 1.7. provide train loss
            if self.config['wandb']['enable']:
                # 1.7.1. upload train loss by wandb
                wandb.log({"train_loss": loss.item()})
        # 1.7.2. print train loss
        print(f"train_loss: {total_loss/len(self.train_dataloader)}")

        # 2. validate model
        self._valid_epoch(epoch)


    def _valid_epoch(self, epoch):

        self.model.eval()
        total_loss = 0
        outputs, targets = [], []
        with torch.no_grad():
            for (data, target) in tqdm(self.valid_dataloader, total=len(self.valid_dataloader), desc=f"epoch {epoch} validing"):
                # 1.1. prepare data
                data, target = data.to(self.device), target.to(self.device)
                # 1.2. run model
                output = self.model(data)
                # 1.3. calculate loss
                loss = self.criterion(output, target)
                # 1.4. save loss & outputs & targets
                total_loss += loss.item()
                outputs.append(output)
                targets.append(target)

        # 2.1. calculate avg loss & matrics
        outputs = torch.cat(outputs)
        targets = torch.cat(targets)
        result = {}
        result["val_loss"] = total_loss/len(self.valid_dataloader)
        for metric in self.metric_ftns:
            result[f"val_{metric.__name__}"] = metric(outputs, targets)

        # 2.2. provide result
        if self.config['wandb']['enable']:
            # 2.2.1 upload result by wandb
            wandb.log(result)
        # 2.2.2. print result
        print(", ".join(f'{key}: {value}'for key, value in result.items()))

        # 2.3. save model if model break best score
        if self.mode == "min" and result[f"val_{self.config['metrics'][0]}"] < self.best_score:
            self.best_score = result[f"val_{self.config['metrics'][0]}"]
            self.save(epoch)

            torch.save(self.model, self.save_file)
        if self.mode == "max" and result[f"val_{self.config['metrics'][0]}"] > self.best_score: 
            self.best_score = result[f"val_{self.config['metrics'][0]}"]
            self.save(epoch)

    def save(self, epoch):
        torch.save({ 'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config,
            f'val_{self.config["metrics"][0]}': self.best_score,
        }, f"{self.save_file}_val_{self.config['metrics'][0]}={self.best_score}.pth")