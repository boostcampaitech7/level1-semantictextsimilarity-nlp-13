import torch
import pandas as pd
from abc import *

class BaseDataset(torch.utils.data.Dataset):
    def __init__(self, data_path, tokenizer, col_info):
        """
        Args:
            data_path (str): csv data file path
            tokenizer : tokenizer
            col_info (dict): csv column information
        """
        super().__init__()
        data = pd.read_csv(data_path)
        self.tokenizer = tokenizer
        self.col_info = col_info
        self.inputs, self.targets = self.preprocessing(data)

    def __getitem__(self, idx):
        if len(self.targets) == 0:
            return torch.tensor(self.inputs[idx])
        else:
            return torch.tensor(self.inputs[idx]), torch.tensor(self.targets[idx])

    def __len__(self):
        return len(self.inputs)

    @abstractmethod
    def preprocessing(self, data):
        """
        Args:
            data (pandas.core.frame.DataFrame): DataFrame 객체로 구성된 dataset
        Returns:
            inputs (List[List[int]]): input list (each input is int list)
            targets (List[float] or List[int]): target list (each target is float(regression) or int(classification))
        """
        pass