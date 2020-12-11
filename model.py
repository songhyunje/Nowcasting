import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F

from ConvLSTM import ConvLSTM


class LightningModel(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams
        self.convLSTM = ConvLSTM(1, self.hparams.num_filters, kernel_size=(7, 7), num_layers=2,
                                 return_all_layers=False)


        if self.hparams.num_filters > 1:
            # self.linear = nn.Linear(self.hparams.num_filters * 1200 * 960, 1200 * 960)
            self.linear = nn.Conv2d(in_channels=self.hparams.num_filters,
                                    out_channels=1,
                                    kernel_size=(7, 7),
                                    padding=3)

    def forward(self, x):
        outputs, _ = self.convLSTM(x)
        if hasattr(self, 'linear'):
            output = self.linear(outputs[:, -1, :, :, :])
        else:
            output = outputs[:, -1, :, :, :]
        return output

    def infer(self, x):
        outputs, _ = self.convLSTM(x)
        if hasattr(self, 'linear'):
            output = self.linear(outputs[:, -1, :, :, :])
        else:
            output = outputs[:, -1, :, :, :]

        return output

    def training_step(self, batch, batch_idx):
        # seqs, target, mask = batch
        seqs, target, mask = batch
        pred = self(seqs)

        pred.masked_fill_(mask, 0)
        target.masked_fill_(mask, 0)
        # loss = F.mse_loss(pred, target)
        loss = F.mse_loss(pred, target, reduction='sum')
        self.log('train_loss', loss, on_epoch=True)
        return {'loss': loss}

    def validation_step(self, batch, batch_idx):
        # seqs, target, mask = batch
        seqs, target, mask = batch
        pred = self(seqs)

        pred.masked_fill_(mask, 0)
        target.masked_fill_(mask, 0)
        # loss = F.mse_loss(pred, target)
        loss = F.mse_loss(pred, target, reduction='sum')
        self.log('val_loss', loss)

    def test_step(self, batch, batch_idx):
        # seqs, target, mask = batch
        seqs, target = batch
        pred = self(seqs)
        mse = F.mse_loss(pred, target)
        # masked_pred = torch.masked_select(pred, mask)
        # masked_target = torch.masked_select(target, mask)
        # mse = F.mse_loss(masked_pred, masked_target)
        self.log('mse', mse)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr, eps=self.hparams.adam_epsilon)
        return optimizer

    @staticmethod
    def add_model_specific_args(parser):
        parser.add_argument("--lr", default=1e-3, type=float, help="The initial learning rate")
        parser.add_argument("--num_filters", default=6, type=int)
        parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
        parser.add_argument("--max_epochs", default=100, type=int, help="Number of training epochs")
        parser.add_argument("--train_batch_size", default=8, type=int, help="Training batch size")
        parser.add_argument("--valid_batch_size", default=2, type=int, help="Valdiation batch size")
        parser.add_argument("--test_batch_size", default=2, type=int, help="Test batch size")
