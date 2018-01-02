# Real Time User-guided Colorizationpy with Learned Deep Priors implemented in pytorch

This is a pytorch implementation of ["Real-Time User-Guided Image Colorization with Learned Deep Priors"](https://arxiv.org/abs/1705.02999) by Zhang et.al.

## Getting Started

### Prerequisites

torch==0.2.0.post4, torchvision==0.1.9

### Installing and running the tests

Make sure you have cifar10 or CelebA downloaded in ./data.
You can download it through by taking a look at my "download.sh" file
```
./data/CelebA
./data/Cifar10
./data/pts_in_hull.npy
```

first clone this repository

```
git clone https://github.com/sjooyoo/https://github.com/sjooyoo/real-time-user-guided-colorization_pytorch.git
```
then run train

```
python deep_color.py
```

### Results
<img src="https://user-images.githubusercontent.com/32257532/34465590-89888210-eef6-11e7-93fe-f061c62c8ef5.png" width="300">

### Note
This is not a complete implementation. I have implemented the global hints network but have yet to incorporate it into the main network.


### Further work


## Acknowledgments
Original paper ["Real-Time User-Guided Image Colorization with Learned Deep Priors"](https://arxiv.org/abs/1705.02999)
