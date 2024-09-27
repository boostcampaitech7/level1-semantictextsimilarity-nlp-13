import torch
import torchmetrics


def accuracy(output, target):
    with torch.no_grad():
        pred = torch.argmax(output, dim=1)
        assert pred.shape[0] == len(target)
        correct = 0
        correct += torch.sum(pred == target).item()
    return correct / len(target)

def pearson(output, target):
    with torch.no_grad():
        return torchmetrics.functional.pearson_corrcoef(output.squeeze(), target.squeeze())