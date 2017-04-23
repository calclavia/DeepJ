import torch

def one_hot(index, size):
    x = torch.zeros(size)
    x[index] = 1
    return x
