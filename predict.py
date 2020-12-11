import argparse
from pathlib import Path

import torch
import numpy as np
from torch.utils.data import DataLoader

from dataset import NowCastingPredctionDataset
from model import LightningModel
import torch.nn.functional as F

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def prediction(args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    prediction_dataset = NowCastingPredctionDataset(args.data_dir, transform=True)
    # collate_fn is needed if you want to put batch_size parameters
    prediction_dataloader = DataLoader(prediction_dataset)
    model = LightningModel.load_from_checkpoint(args.checkpoint)
    model.to(device)
    model.freeze()
    model.eval()

    with torch.no_grad():
        for data in prediction_dataloader:
            fn, seqs, target, mask = data['fn'], data['seqs'], data['target'], data['mask']
            seqs = seqs.to(device)
            mask = mask.to(device)
            target = target.to(device)

            pred = model.infer(seqs)
            # print(F.mse_loss(pred, target, reduction='sum'))

            pred *= 250
            pred[pred < 1] = 0
            # print(pred)
            # pred = torch.where(mask < 0, mask, pred)
            pred.masked_fill_(mask, -9999)
            # print(pred)
            # print(torch.sum(pred[pred > 0]))
            # print(torch.sum(target[target > 0]))

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
