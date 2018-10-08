# Tracking via Colorization
[![KakaoBrain](https://img.shields.io/badge/kakao-brain-ffcd00.svg)](http://kakaobrain.com/)
[![tensorflow](https://img.shields.io/badge/tensorflow-1.10-ed6c20.svg)](https://www.tensorflow.org/)
[![CodeFactor](https://www.codefactor.io/repository/github/wbaek/tracking_via_colorization/badge)](https://www.codefactor.io/repository/github/wbaek/tracking_via_colorization)
[![CircleCI](https://circleci.com/gh/wbaek/tracking_via_colorization.svg?style=svg)](https://circleci.com/gh/wbaek/tracking_via_colorization)

* colorization sample
![tracking via colorization sample2](./datas/samples/sample2.gif)

* tracking segmentation sample
![tracking via colorization sample0](./datas/samples/sample0.gif)
![tracking via colorization sample1](./datas/samples/sample1.gif)


## Introduction

This TensorFlow implementation is designed with these goals:
- [ ] **Tracking via Colorization** Tensorflow implementatin of [Tracking Emerges by Colorizing Videos](https://arxiv.org/abs/1806.09594)



## How to Use

### Clustering
```
python3 bin/clustering.py -k 16 -n 10000 -o datas/centroids/centroids_16k_cifar10_10000samples.npy
```

### Train

* colorizer
```
python3 bin/train_colorizer.py --model-dir models/colorizer
tensorboard --host 0.0.0.0 --port 6006 --logdir models
```

* cifar10
```
python3 bin/train_estimator_cifar10.py --model-dir models/test
```

### Predict

* colorizer
```
python3 bin/test_colorizer.py --checkpoint models/test/model.ckpt-100000 --scale 1 --name davis -o results/davis/
```

### Prerequisite

Should install below libraries.

- Tensorflow >= 1.10
- opencv >= 3.0

And install below dependencies.

```bash
apt install -y libsm6 libxext-dev libxrender-dev libcap-dev
apt install -y ffmpeg
pip install -r requirements.txt
```

