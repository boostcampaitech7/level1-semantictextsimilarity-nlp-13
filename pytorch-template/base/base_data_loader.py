import transformers
import torch
from module import dataset
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

def collate_fn(batch, is_train=True):
    data = []
    targets = []

    for item in batch:
        if isinstance(item, tuple):  # train 데이터의 경우
            data_tensor = item[0]
            target_tensor = item[1]
        else:  # test 데이터의 경우
            data_tensor = item
            target_tensor = None

        data.append(data_tensor)
        if is_train and target_tensor is not None:
            targets.append(target_tensor)

    data_padded = pad_sequence(data, batch_first=True, padding_value=0)

    if is_train:
        targets = torch.stack(targets) if targets else None
        return data_padded, targets
    else:
        return data_padded  # test 데이터에서는 target 없이 데이터만 반환