import pytorch_lightning as pl
from torch.utils.data import DataLoader
import argparse
from dataset import NowCastingDataset


class NowCastingDataModule(pl.LightningDataModule):
    def __init__(self, params):
        super().__init__()
        self.data_dir = params.data_dir
        self.train_batch_size = params.train_batch_size
        self.valid_batch_size = params.valid_batch_size

        if hasattr(params, 'test_batch_size'):
            self.test_batch_size = params.test_batch_size

    def prepare_data(self):
        # Download, tokenize, etc
        # Write to disk or that need to be done only from a single GPU in distributed settings
        NowCastingDataset(self.data_dir, data_type='train', prepare=True)
        NowCastingDataset(self.data_dir, data_type='valid', prepare=True)
        NowCastingDataset(self.data_dir, data_type='test', prepare=True)

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            self.train = NowCastingDataset(self.data_dir, data_type='train', transform=True)
            self.valid = NowCastingDataset(self.data_dir, data_type='valid', transform=True)

        if stage == 'test' or stage is None:
            self.test = NowCastingDataset(self.data_dir, data_type='test')

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.train_batch_size, shuffle=True, num_workers=4)

    def val_dataloader(self):
        return DataLoader(self.valid, batch_size=self.valid_batch_size, num_workers=4)

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=self.test_batch_size)


if __name__ == "__main__":
    args = argparse.Namespace()
    args.data_dir = 'data'
    args.train_batch_size = 2
    args.valid_batch_size = 2

    data_module = NowCastingDataModule(args)
    data_module.prepare_data()
    data_module.setup('fit')
    for batch in data_module.train_dataloader():
        print(batch[0].size())
        print(batch[1].size())
