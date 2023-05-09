## Requirements 
### Environment setup
```
sudo apt-get update
sudo apt-get install software-properties-common build-essential curl git ffmpeg
conda env create -f environment.yml -n <env_name>
```
### Data preperation
```
python prepare_data.py
```
## Training Commands
```
python train_adv_ensemble.py
```
