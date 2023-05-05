## Requirements 
### Environment setup
```
sudo apt-get update
sudo apt-get install build-essential libgl1-mesa-glx
conda env create -f environment.yml
conda activate adv
```
### Data preperation
```
python prepare_data.py
```
## Training Commands
```
python train_adv_ensemble.py
```
