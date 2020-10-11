import argparse

import pytorch_lightning as pl

from data_module import NowCastingDataModule
from model import LightningModel


def train(args):
    # if os.path.exists(args.output_dir) and os.listdir(args.output_dir) \
    #         and args.do_train and not args.overwrite_output:
    #     raise ValueError("Output directory ({}) already exists and is not empty.".format(args.output_dir))
    #
    # checkpoint_callback = pl.callbacks.ModelCheckpoint(
    #     filepath=args.output_dir,
    #     prefix="checkpoint",
    #     monitor="val_loss",
    #     verbose=True,
    #     mode="min",
    #     save_top_k=3
    # )

    data_module = NowCastingDataModule(data_dir=args.data_dir, batch_size=2)
    model = LightningModel(args)
    # trainer = pl.Trainer(gpus=2, distributed_backend='dp', max_epochs=100)
    trainer = pl.Trainer(gpus=1, max_epochs=100)

    trainer.fit(model, data_module)

    # trainer.test()
    # if args.do_predict:
    #     trainer.test()


def add_generic_arguments(parser):
    parser.add_argument("--data_dir", default=None, type=str, required=True,
                        help="The input data dir.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_generic_arguments(parser)
    args = parser.parse_args()
    train(args)
