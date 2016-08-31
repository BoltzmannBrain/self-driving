# self-driving

Machine learning models for self-driving cars. This repo is a WIP and will be updated regularly.

Requires:

* [Keras](https://github.com/fchollet/keras/)
* [TensorFlow](https://www.tensorflow.org/versions/r0.10/get_started/os_setup.html)
* [OpenCV](http://opencv.org/downloads.html)

## Models and Tasks

#### Prediction with deep learning

The specific task here is to predict driving speed from a dashboard-mounted camera; there's tons of this data open-sourced by the folks at comma.ai [here](https://github.com/commaai/research). I experimented with several flavors of deep nets:

1. **Static CNN** -- The model is based off of NVIDIA's ["End-to-end Self-driving" model](https://arxiv.org/abs/1604.07316). It's *static* because each video frame is assumed independent by the model, ignoring temporal dependencies. The resulting model is in models/cnn_prediction.py.
2. **CNN + LSTM** -- In order to incorporate temporal info into the static CNN, I experimented with several architectures discussed in ["Beyond Short Snippets: Deep Networks for Video Classification"](http://arxiv.org/abs/1503.08909). The resulting model is in models/cnn_lstm_prediction.py.
3. **Optical flow** -- Tracking feature points with sparse optical flow implicitly encodes temporal dependencies between frames. (This model will be pushed soon!)

Check out [speed_prediction.md](speed_prediction.md) for more.

![](training_screenshot.png?raw=true "Screenshot from training" =200x)



## Datasets

A few places to find open-source data to play with:

* https://www.cityscapes-dataset.com/
* http://robotcar-dataset.robots.ox.ac.uk/examples/
* http://selfracingcars.com/blog/2016/7/26/polysync
* http://data.selfracingcars.com/
* http://research.comma.ai/