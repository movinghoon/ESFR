## Early-Stage Feature Reconstruction (ESFR)
Dong Hoon Lee, Sae-Young Chung, Unsupervised Embedding Adaptation via Early-Stage Feature Reconstruction for Few-Shot Classification, ICML 2021



### Dependencies

This code requires the following:

```
tensorflow >= 2.0.0
numpy
tqdm
```



### Usage

For details, see the arguments inside `run.py`

* Testing BD-CSPN + ESFR on mini-ImageNet/WidResNet/1-shot

```
python run.py --shot 1 --dataset mini --architecture WidResNet
```

