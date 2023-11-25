import torch.utils.data as data
import torch


class ImgFile(data.Dataset):
    def __init__(self, I, T, L, M1, M2):
        self.images = torch.from_numpy(I)
        self.tags = torch.from_numpy(T)
        self.labels = torch.from_numpy(L)
        self.m1s = torch.from_numpy(M1)
        self.m2s = torch.from_numpy(M2)
        self.length = L.shape[0]

    def __getitem__(self, index):
        img = self.images[index]
        tag = self.tags[index]
        label = self.labels[index]
        m1 = self.m1s[index]
        m2 = self.m2s[index]

        return img, tag, label, m1, m2

    def __len__(self):
        return self.length

