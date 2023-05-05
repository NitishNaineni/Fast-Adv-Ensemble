## Requirements 
### Environment setup
```
conda env create -f environment.yml
```
### Data preperation
for some of the dependecies gcc is needed. run `sudo apt install build-essential` if gcc isnt installed
```
python prepare_data.py
```
## Training Commands
```
python train_adv_ensemble.py
```
