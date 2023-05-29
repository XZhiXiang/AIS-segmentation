# AIS-segmentation


## Requirements
CUDA 11.0<br />
Python 3.7<br /> 
Pytorch 1.7<br />
Torchvision 0.8.2<br />

## Usage

### 0. Installation
* Install Pytorch1.7, nnUNet and AISD as below
  
```

pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html

cd nnUNet
pip install -e .

cd package
pip install -e .

pip install matplotlib
pip install batchgenerators==0.21
```

### 1. Training 
cd package/AISD/run

* Run `nohup python run_training.py -gpu='0' -outpath='AISD' 2>&1 &` for training.

### 2. Testing 
* Run `nohup python run_training.py -gpu='0' -outpath='AISD' -val --val_folder='validation_output' 2>&1 &` for validation.


```

