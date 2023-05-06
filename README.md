## Requirements 
### Environment setup
```
sudo apt-get update
sudo apt-get install libopencv-dev python3-opencv libturbojpeg0-dev build-essential libgl1-mesa-glx
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
conda install cupy pkg-config libjpeg-turbo opencv numba cudatoolkit cudnn matplotlib scipy=1.7.3
pip install ffcv composer torchattacks==3.2.7
```
### Data preperation
```
python prepare_data.py
```
## Training Commands
```
python train_adv_ensemble.py
```
