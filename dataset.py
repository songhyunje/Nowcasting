from pathlib import Path
import os

import netCDF4 as nc
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset


class NowCastingDataset(Dataset):
    def __init__(self, data_dir: str = './data', data_type: str = 'train',
                 prepare=False, seq_len: int = 6):
        super().__init__()
        self.data_dir = data_dir
        self.seq_len = seq_len

        if prepare:
            self.prepare(data_dir, data_type)

        tmp = torch.load(os.path.join(self.processed_folder, data_type))
        self.data = [tmp[i:i+self.seq_len] for i in range(len(tmp)+1-self.seq_len)]

    def prepare(self, data_dir, data_type):
        if self._check_exists(data_type):
            return

        print('Processing...')
        tmp = []
        path = Path(data_dir + '/' + data_type)
        for x in path.iterdir():
            if x.is_dir():
                files = sorted([f for f in x.iterdir() if f.is_file()])
                for f in files:
                    ds = nc.Dataset(f)
                    value = ds.variables["data"][:]
                    tmp.append(torch.tensor(value))

        with open(os.path.join(self.processed_folder, data_type), 'wb') as f:
            torch.save(tmp, f)

        print('Done!')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        item = torch.stack(self.data[index])
        mask = item[-1].ge(0)
        return item[:-1], item[-1], mask
        # mask = item[1:].ge(0)
        # return item[:-1], item[1:], mask

    @property
    def processed_folder(self):
        return os.path.join(self.data_dir, 'processed')

    def _check_exists(self, data_type):
        return os.path.exists(os.path.join(self.processed_folder, data_type))


class NowCastingPredctionDataset(Dataset):
    def __init__(self, data_dir: str = './data'):
        super().__init__()

        self.data = []
        path = Path(data_dir)
        for x in path.iterdir():
            if x.is_dir():
                files = sorted([f for f in x.iterdir() if f.is_file()])
                tmp = []
                for f in files:
                    ds = nc.Dataset(f)
                    value = ds.variables["data"][:]
                    tmp.append(torch.tensor(value))
                self.data.append((x, tmp))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        fn, data = self.data[index]
        item = torch.stack(data)
        return {'fn': str(fn), 'seqs': item[:-1], 'target': item[-1]}


if __name__ == "__main__":
    dataset = NowCastingDataset(data_type='train')
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
    for d in dataloader:
        print(d[0].size())
        print(d[1].size())
        print(d[2].size())
        break
