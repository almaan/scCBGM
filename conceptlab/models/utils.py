import torch.nn.functional as F


def sigmoid(x, alpha=1):
    return F.sigmoid(alpha * x)
