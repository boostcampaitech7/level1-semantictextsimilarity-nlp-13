import transformers
import torch
from module import dataset


class DataModule:
    def __init__(self, plm_name, dataset_name, batch_size, shuffle, train_path, dev_path, test_path, predict_path, col_info, max_length):
        super().__init__()
        self.plm_name = plm_name
        self.dataset_name = dataset_name
        self.batch_size = batch_size
        self.shuffle = shuffle

        self.train_path = train_path
        self.dev_path = dev_path
        self.test_path = test_path
        self.predict_path = predict_path

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.predict_dataset = None

        self.tokenizer = transformers.AutoTokenizer.from_pretrained(plm_name, max_length=max_length)
        self.col_info = col_info
        self.setup()

    def setup(self):
            self.train_dataset = getattr(dataset, self.dataset_name)(self.train_path, self.tokenizer, self.col_info)
            self.val_dataset = getattr(dataset, self.dataset_name)(self.dev_path, self.tokenizer, self.col_info)
            self.test_dataset = getattr(dataset, self.dataset_name)(self.test_path, self.tokenizer, self.col_info)
            self.predict_dataset = getattr(dataset, self.dataset_name)(self.predict_path, self.tokenizer, self.col_info)

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=self.shuffle)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val_dataset, batch_size=self.batch_size)

    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.test_dataset, batch_size=self.batch_size)

    def predict_dataloader(self):
        return torch.utils.data.DataLoader(self.predict_dataset, batch_size=self.batch_size)