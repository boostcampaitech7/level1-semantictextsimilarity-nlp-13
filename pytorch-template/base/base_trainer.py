import torch
import os
from abc import abstractmethod
import wandb


class BaseTrainer:
    """
    Base class for all trainers
    """

    def __init__(self, model, criterion, metric_ftns, optimizer, config):
        self.config = config

        self.model = model
        self.criterion = criterion
        self.metric_ftns = metric_ftns
        self.optimizer = optimizer

        cfg_trainer = config["trainer"]
        self.epochs = cfg_trainer["epochs"]
        self.mode = config["trainer"]["mode"]
        self.best_score = float("inf") if self.mode == "min" else 0
        if not os.path.exists(config["trainer"]["save_dir"]):
            os.makedirs(config["trainer"]["save_dir"])
        self.save_file = (
            f'{config["trainer"]["save_dir"]}'
            + f'{self.config["arch"]["type"]}_{self.config["arch"]["args"]["plm_name"]}'.replace(
                "/", "-"
            )
        )

    @abstractmethod
    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Current epoch number
        """
        raise NotImplementedError

    def train(self):
        """
        Full training logic
        """
        if self.config['wandb']['enable']:
            wandb.init(
                project=self.config["wandb"]["project_name"],
                name=self.save_file.split("/")[-1],
            )
        for epoch in range(self.epochs + 1):
            self._train_epoch(epoch)
