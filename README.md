# Tracking via Colorization
[![KakaoBrain](https://img.shields.io/badge/kakao-brain-ffcd00.svg)](http://kakaobrain.com/)
[![tensorflow](https://img.shields.io/badge/tensorflow-1.10-ed6c20.svg)](https://www.tensorflow.org/)
[![CodeFactor](https://www.codefactor.io/repository/github/wbaek/tracking_via_colorization/badge)](https://www.codefactor.io/repository/github/wbaek/tracking_via_colorization)
[![CircleCI](https://circleci.com/gh/wbaek/tracking_via_colorization.svg?style=svg)](https://circleci.com/gh/wbaek/tracking_via_colorization)


## Introduction

This TensorFlow implementation is designed with these goals:
- [ ] **Tracking via Colorization** Tensorflow implementatin of [Tracking Emerges by Colorizing Videos](https://arxiv.org/abs/1806.09594)



## How to Use

### Train

* cifar10
```
python3 bin/train_estimator_cifar10.py --model-dir models/test
tensorboard --host 0.0.0.0 --port 6006 --logdir models

```

### Prerequisite

Should install below libraries.

- Tensorflow >= 1.10
- opencv >= 3.0

And install below dependencies.

```bash
apt install -y libsm6 libxext-dev libxrender-dev
pip install -r requirements.txt
```

