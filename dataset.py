import netCDF4 as nc
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

from pathlib import Path


class NowCastingDataset(Dataset):
    def __init__(self, data_dir: str = './data', data_type: str = 'train'):
        super().__init__()

        self.data = []
        path = Path(data_dir + '/' + data_type)
        for x in path.iterdir():
            if x.is_dir():
                files = sorted([f for f in x.iterdir() if f.is_file()])
                tmp = []
                for f in files:
                    ds = nc.Dataset(f)
                    value = ds.variables["data"][:]
                    tmp.append(torch.tensor(value))
                self.data.append(torch.stack(tmp))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index][:-1], self.data[index][-1]


if __name__ == "__main__":
    dataset = NowCastingDataset(data_type='train')
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
    for d in dataloader:
        print(d[0].size())
        print(d[1].size())
        break
