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
   
   
