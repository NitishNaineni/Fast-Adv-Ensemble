## Requirements 
### Environment setup
for some of the dependecies gcc is needed. run `sudo apt install build-essential` if gcc isnt installed
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
