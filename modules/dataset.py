import numpy as np
from torch.utils.data import Dataset


class ImageDataset(Dataset):
    def __init__(self, feature_arr, label_arr, pre_process=True):
        self.feature_arr = feature_arr
        self.label_arr = label_arr
        self.pre_process = pre_process

    def __len__(self):
        return self.feature_arr.shape[0]

    def __getitem__(self, idx):
        if self.pre_process:
            im = self.feature_arr[idx, 0].astype(np.float)
            im -= im.mean()
            im[im > 0] /= im.max()
            im[im < 0] /= -im.min()
        else:
            im = self.feature_arr[idx, 0]
        new_shape = (1, ) + im.shape
        im = im.reshape(new_shape)
        return im, self.label_arr[idx]
