import argparse
from pathlib import Path

import torch
import numpy as np
from torch.utils.data import DataLoader

from dataset import NowCastingPredctionDataset
from model import LightningModel

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def prediction(args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    prediction_dataset = NowCastingPredctionDataset(args.data_dir)
    # collate_fn is needed if you want to put batch_size parameters
    prediction_dataloader = DataLoader(prediction_dataset)
    model = LightningModel.load_from_checkpoint(args.checkpoint)
    model.to(device)
    model.freeze()
    model.eval()

    with torch.no_grad():
        for data in prediction_dataloader:
            fn, seqs, mask = data['fn'], data['seqs'], data['mask']
            seqs = seqs.to(device)
            mask = mask.to(device)
            # target = data['target'].to(device)
            # target = target.to(device)
            pred = model.infer(seqs)
            pred = torch.where(mask < 0, mask, pred)

            path = Path(args.output_dir + '/' + fn[0].replace('/', '_'))
            np.save(path, pred[0].cpu().numpy())


def add_generic_arguments(parser):
    parser.add_argument("--data_dir", default=None, type=str, required=True)
    parser.add_argument("--output_dir", default=None, type=str, required=True)
    parser.add_argument("--checkpoint", default=None, type=str, required=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_generic_arguments(parser)
    args = parser.parse_args()
    prediction(args)
