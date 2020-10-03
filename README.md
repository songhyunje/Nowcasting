# Nowcasting using ConvLSTM

PyTorch and PyTorch-Lightning implementation for Nowcasting

### Dependency
- PyTorch >= 1.6
- pytorch-lightning >= 0.9
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
python trainer.py --data_dir data
```

## TODO (in progress)
- Modify kernel size 
- Process -9999 value
- Add some layers for the prediction
- ...

## Acknowledgement
The ConvLSTM implementation is inspired from [ConvLSTM](https://github.com/ndrplz/ConvLSTM_pytorch).

