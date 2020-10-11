import pytorch_lightning as pl
from torch.utils.data import DataLoader

from dataset import NowCastingDataset


class NowCastingDataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str = './data', batch_size=16):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size

    def prepare_data(self):
        # Download, tokenize, etc
        # Write to disk or that need to be done only from a single GPU in distributed settings
        NowCastingDataset(self.data_dir, data_type='train', prepare=True)
        NowCastingDataset(self.data_dir, data_type='valid', prepare=True)
        NowCastingDataset(self.data_dir, data_type='test', prepare=True)

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            self.train = NowCastingDataset(self.data_dir, data_type='train')
            self.valid = NowCastingDataset(self.data_dir, data_type='valid')

        if stage == 'test' or stage is None:
            self.test = NowCastingDataset(self.data_dir, data_type='test')

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.valid, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=self.batch_size)


if __name__ == "__main__":
    data_module = NowCastingDataModule()
    data_module.prepare_data()
    data_module.setup('fit')
    for batch in data_module.train_dataloader():
        print(batch[0].size())
        print(batch[1].size())
