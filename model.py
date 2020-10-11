import pytorch_lightning as pl
import torch
import torch.nn.functional as F

from ConvLSTM import ConvLSTM


class LightningModel(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams
        self.convLSTM = ConvLSTM(1, 1, kernel_size=(5, 5), num_layers=3,
                                 return_all_layers=False)

    def forward(self, x):
        outputs, _ = self.convLSTM(x)
        # return outputs[:, -1, :, :, :]  # get last hidden
        return outputs  # get last hidden

    def training_step(self, batch, batch_idx):
        seqs, target, mask = batch
        pred = self(seqs)
        masked_pred = torch.masked_select(pred, mask)
        masked_target = torch.masked_select(target, mask)
        loss = F.mse_loss(masked_pred, masked_target)
        # result = pl.TrainResult(loss)
        # result.log('train_loss', loss, on_epoch=True)
        self.log('train_loss', loss, on_epoch=True)
        return {'loss': loss}

    def validation_step(self, batch, batch_idx):
        seqs, target, mask = batch
        pred = self(seqs)
        masked_pred = torch.masked_select(pred, mask)
        masked_target = torch.masked_select(target, mask)
        loss = F.mse_loss(masked_pred, masked_target)
        # result = pl.EvalResult(checkpoint_on=loss)
        # result.log('loss', loss)
        self.log('val_loss', loss)
        return {'val_loss': loss}

    # def validation_step_end(self, batch_parts):
    #     # do something with both outputs
    #     return torch.mean(batch_parts.loss)

    def test_step(self, batch, batch_idx):
        seqs, labels = batch
        preds = self(seqs)

        mse = F.mse_loss(preds, labels)
        self.log('mse', mse)
        return {'mse': mse}

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr, eps=self.hparams.adam_epsilon)
        return optimizer

    @staticmethod
    def add_model_specific_args(parser):
        parser.add_argument("--lr", default=1e-3, type=float, help="The initial learning rate")
        parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
        parser.add_argument("--max_epochs", default=50, type=int, help="Number of training epochs")
        # parser.add_argument("--train_batch_size", default=2, type=int, help="Number of training epochs")
        # parser.add_argument("--valid_batch_size", default=2, type=int, help="Number of training epochs")
        # parser.add_argument("--test_batch_size", default=2, type=int, help="Number of training epochs")
