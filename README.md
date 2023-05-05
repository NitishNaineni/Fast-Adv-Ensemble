## Requirements 
### Environment setup
ffcv need gcc to build. run `sudo apt install build-essential` if gcc isnt installed

opencv need libgl1-mesa-glx. run `sudo apt-get install libgl1-mesa-glx` if not installed
```
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
