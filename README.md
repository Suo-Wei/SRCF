[![LICENSE](https://img.shields.io/badge/license-MIT-green)](https://github.com/taksau/GPS-Net/blob/master/LICENSE)
[![Python](https://img.shields.io/badge/python-3.7-blue.svg)](https://www.python.org/)
![PyTorch](https://img.shields.io/badge/pytorch-1.4.0-%237732a8) 

# A Simple and Robust Correlation Filtering method for text-based person search
We provide the code for reproducing results of our ECCV 2022 paper [A Simple and Robust Correlation Filtering method for text-based person search](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136950719.pdf). 
Compared with the original paper, we obtain better performance due to some modifications. Following our global response map, we also add the same mutual-exclusion-loss to separate body part response map. Meanwhile, we merge the global filter and dictionary filter module. The adjusted method achieves a new state-of-the-art performance and it improves to 64.88 on Top-1 *without* [Re-Rank](https://github.com/TencentYoutuResearch/PersonReID-NAFS?utm_source=catalyzex.com) (CUHK-PEDES).
## Getting Started
### Requirements
- [PyTorch](https://pytorch.org/) 1.4 or higher
- [transformers](https://huggingface.co/docs/transformers/index) (install with `pip install transformers`)
- numpy, torchvision

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
   cd SRCF
   python ./dataset/process_CUHK_data.py
   ~~~
   
### Building BERT
~~~
mkdir bert_weight
~~~

Downland the [weight and config](https://huggingface.co/bert-base-uncased/tree/main), put them into SRCF/bert_weight
   
### Training and Testing
~~~
bash run/train.bash 
~~~
### Evaluation
~~~
bash run/test.bash
~~~

## Results on CUHK-PEDES

|CUHK-PEDES | performance |
|------|------|
| `Top-1` | 64.88 |
| `Top-5` | 83.02 |
| `Top-10` | 88.56 |

## Citation

If this work is helpful for your research, please cite our work:

~~~
@InProceedings{Suo_ECCV_A,
author = {Suo, Wei and Sun, MengYang and Niu, Kai, et.al},
title = {A Simple and Robust Correlation Filtering method for text-based person search},
booktitle = {The European Conference on Computer Vision (ECCV)},
month = {August},
year = {2022}
}
~~~

### References
[SSAN](https://github.com/zifyloo/SSAN/)
