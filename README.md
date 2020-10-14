# Nowcasting using ConvLSTM

PyTorch and PyTorch-Lightning implementation for Nowcasting

### Dependency
- PyTorch >= 1.6
- pytorch-lightning >= 1.0
- netcdf4

Please check the requirements.txt.

```bash
pip install -r requirements.txt
```

## Getting Started
### Step 1: Prepare the data
Before training or evaluation, we should prepare the data directory as follows:
```
data
└── train
    └── instance1
        └── data_00001.nc
        └── data_00002.nc
        └── ...
    └── instance2 
        └── data_00001.nc
        └── data_00002.nc
        └── ...
    └── ...
└── valid
    └── instance1
        └── data_00001.nc
        └── data_00002.nc
        └── ...
    └── instance2 
        └── data_00001.nc
        └── data_00002.nc
        └── ...
    └── ...
└── test 
    └── instance1
        └── data_00001.nc
        └── data_00002.nc
        └── ...
    └── instance2 
        └── data_00001.nc
        └── data_00002.nc
        └── ...
    └── ...
```

instance x is regarded as one data. The last file in the instance x directory will be the target.
Please check dataset.py. 

### Step 2: Train the model
Here is a command line example of the code:

```bash
python trainer.py --data_dir $DATA_DIR --output_dir $OUTPUT_DIR --gpus 2 --precision 16 --accumulate_grad_batches 5 --gradient_clip_val 1.0
```

You can change the parameters. Please check arguments of model.py and trainer.py.

## Prediction using Pretrained model
1. Download [pretrained model](https://drive.google.com/file/d/1PNA0Ywxbo1poX_5KL7Cv-HkZfUvWJWL8/view?usp=sharing).

2. Put the downloaded model at $CHECKPOINT_DIR

3. Prepare the data for the prediction and put it at $DATA_DIR

4. Predict!. Here is an example execution. 
```bash
python predict.py --data_dir data/test --checkpoint output/model.ckpt --output_dir prediction
```

## TODO (in progress)
- Modify kernel size using validation dataset
- Add some layers for the prediction

## Acknowledgement
The ConvLSTM implementation is inspired from [ConvLSTM_pytorch](https://github.com/ndrplz/ConvLSTM_pytorch).

