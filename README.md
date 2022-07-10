[![LICENSE](https://img.shields.io/badge/license-MIT-green)](https://github.com/taksau/GPS-Net/blob/master/LICENSE)
[![Python](https://img.shields.io/badge/python-3.7-blue.svg)](https://www.python.org/)
![PyTorch](https://img.shields.io/badge/pytorch-1.5.0-%237732a8) 

# A Simple and Robust Correlation Filtering method for text-based person search
We provide the code for reproducing results of our ECCV 2022 paper [A Simple and Robust Correlation Filtering method for text-based person search](www.baidu.com). Compared with the original paper, we obtain better performance through some modifications. Following our global response map, we also add the same mutual-exclusion-loss to separate body part response map. Adjusted method achieve new state-of-the-art performance and it improves to 64.89 on Top-1 (CUHK-PEDES).
## Getting Started
### Dataset Preparation

Organize them in `dataset` folder as follows:
    
   ~~~
   |-- dataset/
   |   |-- <CUHK-PEDES>/
   |       |-- imgs
               |-- cam_a
               |-- cam_b
               |-- ...
   |       |-- reid_raw.json
   
   ~~~
    Download the CUHK-PEDES dataset from [here](https://github.com/ShuangLI59/Person-Search-with-Natural-Language-Description) and then run the `process_CUHK_data.py` as follow:
   ~~~
   cd SSAN
   python ./dataset/process_CUHK_data.py
   ~~~
   
#### Training and Testing
~~~
sh experiments/CUHK-PEDES/train.sh 
sh experiments/ICFG-PEDES/train.sh 
~~~
#### Evaluation
~~~
sh experiments/CUHK-PEDES/test.sh 
sh experiments/ICFG-PEDES/test.sh 
~~~

## Results on CUHK-PEDES and ICFG-PEDES
