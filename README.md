# "Real Time User-guided Colorization with Learned Deep Priors" implemented in pytorch

This is a pytorch implementation of ["Real-Time User-Guided Image Colorization with Learned Deep Priors"](https://arxiv.org/abs/1705.02999) by Zhang et.al.

## Getting Started

### Prerequisites

torch==0.2.0.post4, torchvision==0.1.9
The code is written with the default setting that you have gpu. Cpu mode is not recommended when using this repository.

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

to sample results you first need to run deep_color.py, which will automatically save models under a models folder that will be made in your root directory.
I did not include pretrained models in this repository. The --model unet100.pkl below is a sample after 100 epochs. Change the command according to your model that you want to sample.
```
python sampling.py --model unet100.pkl
```


### Results

Input black and white image

<img src="https://user-images.githubusercontent.com/32257532/34475069-350aaf80-efcb-11e7-8a79-d77a593273be.png" width="250">

Predicted colorization output

<img src="https://user-images.githubusercontent.com/32257532/34475079-686cd448-efcb-11e7-95a5-7deb44c06148.png" width="250">

Ground truth image

<img src="https://user-images.githubusercontent.com/32257532/34475088-914b697e-efcb-11e7-8580-2e624af9842e.png" width="250">


### Note
This is not a complete implementation. I have implemented the global hints network but have yet to incorporate it into the main network.


### Further work
* global hints network


## Acknowledgments
Original paper ["Real-Time User-Guided Image Colorization with Learned Deep Priors"](https://arxiv.org/abs/1705.02999)
