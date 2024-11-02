import os
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
import torch


class IrisDataLoader(Dataset):
    def __init__(self, csv_path, transform=None):
        self.data_path = csv_path
        self.transform = transform
        assert os.path.exists(csv_path), f"{csv_path} does not exist"
        df = pd.read_csv(
            csv_path, names=["sepal length", "sepal width", "petal length", "petal width", "label"])
        d = {"setosa": 0, "versicolor": 1, "virginica": 2}
        df["label"] = df["label"].map(d)

        data = df.iloc[:, 0:4]
        label = df.iloc[:, 4:]

        # 数据均值化 tips
        data = (data - data.mean()) / data.std()

        # to tensor
        self.data = torch.from_numpy(np.array(data, dtype=np.float32))
        self.label = torch.from_numpy(np.array(label, dtype=np.int64))

        self.data_num = len(self.data)
        print("data num:", self.data_num)

    def __len__(self):
        return self.data_num

    def __getitem__(self, index):
        return self.data[index], self.label[index]
