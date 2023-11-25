import torch
import numpy as np
import scipy.io as sio


def calculate_hamming_distance(a, b):
    q = a.shape[-1]
    return 0.5 * (q - torch.mm(a, b.T))


def calculate_s(labels1, labels2):
    s = torch.mm(labels1, labels2.T)
    return s


def normalize(x):
    l2_norm = np.linalg.norm(x, axis=1)[:, None]
    l2_norm[np.where(l2_norm == 0)] = 1e-6
    x = x/l2_norm
    return x


def zero_mean(x, mean_val=None):
    if mean_val is None:
        mean_val = np.mean(x, axis=0)
    x -= mean_val
    return x, mean_val


if __name__ == '__main__':
    pass
