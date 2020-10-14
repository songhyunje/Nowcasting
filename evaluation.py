import argparse

import pytorch_lightning as pl
from torch.utils.data import DataLoader

from dataset import NowCastingDataset
from model import LightningModel


def evaluation(args):
    test_dataset = NowCastingDataset(args.data_dir, data_type='test')
    test_dataloader = DataLoader(test_dataset, batch_size=2, num_workers=4)
    model = LightningModel.load_from_checkpoint(args.checkpoint)
    model.freeze()
    trainer = pl.Trainer.from_argparse_args(args)
    trainer.test(model, test_dataloaders=test_dataloader)


def add_generic_arguments(parser):
    parser.add_argument("--data_dir", default=None, type=str, required=True)
    parser.add_argument("--checkpoint", default=None, type=str, required=True)
    parser.add_argument("--gpus", type=int)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_generic_arguments(parser)
    args = parser.parse_args()
    evaluation(args)
