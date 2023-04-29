## Installation
```
conda create -y -n fast python=3.9 cupy pkg-config libjpeg-turbo opencv pytorch torchvision cudatoolkit=11.3 numba -c pytorch -c conda-forge
conda activate fast
pip install ffcv
```