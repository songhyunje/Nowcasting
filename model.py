import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from pytorch_lightning.metrics.regression import MSE

from ConvLSTM import ConvLSTM


class LightningModel(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams
        self.convLSTM = ConvLSTM(1, 1, kernel_size=(3, 3), num_layers=3,
                                 return_all_layers=False)

    def forward(self, x):
        outputs, _ = self.convLSTM(x)
        return outputs[:, -1, :, :, :]  # get last hidden

    def training_step(self, batch, batch_idx):
        seqs, labels = batch
        preds = self(seqs)
        loss = F.mse_loss(preds, labels)
        result = pl.TrainResult(loss)
        result.log('train_loss', loss, on_epoch=True)
        return result

    def validation_step(self, batch, batch_idx):
        seqs, labels = batch
        preds = self(seqs)
        loss = F.mse_loss(preds, labels)
        result = pl.EvalResult(checkpoint_on=loss)
        # {'loss': loss, 'acc': acc, ..., 'metric_n': metric_n}
        # result.log_dict(values)
        result.log('loss', loss)
        return result

    def validation_step_end(self, batch_parts):
        # do something with both outputs
        return torch.mean(batch_parts.loss)

    def test_step(self, batch, batch_idx):
        seqs, labels = batch
        preds = self(seqs)

        mse = MSE(preds, labels)
        result = pl.EvalResult()
        result.log('mse', mse)
        return result

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
